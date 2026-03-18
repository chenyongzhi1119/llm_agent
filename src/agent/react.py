"""Hand-written ReAct Agent core logic"""

import uuid
from openai import OpenAI
from loguru import logger

from config.settings import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, MAX_REACT_STEPS
from src.agent.prompt import REACT_SYSTEM_PROMPT, REACT_USER_PROMPT
from src.agent.parser import parse_react_output, ReActAction, ReActFinish
from src.tools.base import ToolRegistry
from src.memory.short_term import ShortTermMemory
from src.memory.long_term import init_db, save_message, load_history


class ReActAgent:
    """
    Hand-written ReAct Agent with memory support.

    Implements the Thought -> Action -> Observation loop:
    1. LLM thinks and decides the next action
    2. Parse LLM output to extract tool call
    3. Execute tool and get Observation
    4. Feed Observation back to LLM for further reasoning
    5. Repeat until Final Answer or max steps reached

    Memory:
    - Short-term: sliding window keeps recent messages in context
    - Long-term: SQLite persists full conversation history
    """

    def __init__(self, tool_registry: ToolRegistry, session_id: str | None = None):
        self.client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        self.tool_registry = tool_registry
        self.model = LLM_MODEL
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.short_term = ShortTermMemory()

        # Initialize DB and set system prompt
        init_db()
        system_prompt = REACT_SYSTEM_PROMPT.format(
            tools_description=self.tool_registry.get_tools_prompt()
        )
        self.short_term.set_system_prompt(system_prompt)

        # Restore previous conversation from long-term memory
        history = load_history(self.session_id)
        for msg in history:
            self.short_term.add(msg["role"], msg["content"])

        logger.info(f"Session: {self.session_id} | Restored {len(history)} messages from history")

    def run(self, question: str) -> str:
        """
        Execute the ReAct reasoning loop.

        Args:
            question: User question

        Returns:
            Final answer string
        """
        # Add user question to memory
        self.short_term.add("user", REACT_USER_PROMPT.format(question=question))
        save_message(self.session_id, "user", question)

        consecutive_errors = 0
        MAX_CONSECUTIVE_ERRORS = 3

        for step in range(MAX_REACT_STEPS):
            logger.info(f"--- ReAct step {step + 1} ---")

            # Call LLM with current context window
            response = self._call_llm(self.short_term.get_messages())
            logger.info(f"LLM output:\n{response}")

            # Parse output
            parsed = parse_react_output(response)

            if isinstance(parsed, ReActFinish):
                logger.info(f"Final answer: {parsed.answer}")
                self.short_term.add("assistant", parsed.answer)
                save_message(self.session_id, "assistant", parsed.answer)
                return parsed.answer

            if isinstance(parsed, ReActAction):
                logger.info(f"Calling tool: {parsed.tool_name}, input: {parsed.tool_input}")

                observation = self._execute_tool(parsed.tool_name, parsed.tool_input)
                logger.info(f"Tool returned: {observation}")

                # Dead-loop guard: break out if tool keeps erroring
                if observation.startswith("Error:"):
                    consecutive_errors += 1
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        logger.warning(f"Tool '{parsed.tool_name}' failed {consecutive_errors} times in a row, stopping.")
                        answer = f"I was unable to complete the task: {observation}"
                        self.short_term.add("assistant", answer)
                        save_message(self.session_id, "assistant", answer)
                        return answer
                else:
                    consecutive_errors = 0

                self.short_term.add("assistant", response)
                self.short_term.add("user", f"Observation: {observation}")

        # Max steps reached
        logger.warning(f"Reached max steps {MAX_REACT_STEPS}, forcing stop")
        answer = "Sorry, I could not reach a satisfactory answer after multiple reasoning steps. Please try rephrasing your question."
        save_message(self.session_id, "assistant", answer)
        return answer

    def clear_history(self) -> None:
        """Clear conversation history for this session"""
        from src.memory.long_term import clear_history
        self.short_term.clear()
        clear_history(self.session_id)
        logger.info(f"Session {self.session_id}: history cleared")

    def _call_llm(self, messages: list[dict]) -> str:
        """Call LLM and get response"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=512,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """Execute a tool call"""
        tool = self.tool_registry.get(tool_name)
        if tool is None:
            return f"Error: tool '{tool_name}' not found. Available tools: {[t.name for t in self.tool_registry.list_tools()]}"

        try:
            return tool.execute(**tool_input)
        except Exception as e:
            logger.error(f"Tool {tool_name} execution failed: {e}")
            return f"Tool execution error: {e}"
