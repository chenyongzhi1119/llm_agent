"""Hand-written ReAct Agent core logic"""

from openai import OpenAI
from loguru import logger

from config.settings import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, MAX_REACT_STEPS
from src.agent.prompt import REACT_SYSTEM_PROMPT, REACT_USER_PROMPT
from src.agent.parser import parse_react_output, ReActAction, ReActFinish
from src.tools.base import ToolRegistry


class ReActAgent:
    """
    Hand-written ReAct Agent

    Implements the Thought -> Action -> Observation loop:
    1. LLM thinks and decides the next action
    2. Parse LLM output to extract tool call
    3. Execute tool and get Observation
    4. Feed Observation back to LLM for further reasoning
    5. Repeat until Final Answer or max steps reached
    """

    def __init__(self, tool_registry: ToolRegistry):
        self.client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        self.tool_registry = tool_registry
        self.model = LLM_MODEL

    def run(self, question: str) -> str:
        """
        Execute the ReAct reasoning loop.

        Args:
            question: User question

        Returns:
            Final answer string
        """
        system_prompt = REACT_SYSTEM_PROMPT.format(
            tools_description=self.tool_registry.get_tools_prompt()
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": REACT_USER_PROMPT.format(question=question)},
        ]

        steps = []

        for step in range(MAX_REACT_STEPS):
            logger.info(f"--- ReAct step {step + 1} ---")

            # Call LLM
            response = self._call_llm(messages)
            logger.info(f"LLM output:\n{response}")

            # Parse output
            parsed = parse_react_output(response)

            if isinstance(parsed, ReActFinish):
                logger.info(f"Final answer: {parsed.answer}")
                steps.append({"type": "finish", "answer": parsed.answer})
                return parsed.answer

            if isinstance(parsed, ReActAction):
                logger.info(f"Calling tool: {parsed.tool_name}, input: {parsed.tool_input}")

                # Execute tool
                observation = self._execute_tool(parsed.tool_name, parsed.tool_input)
                logger.info(f"Tool returned: {observation}")

                steps.append({
                    "type": "action",
                    "tool": parsed.tool_name,
                    "input": parsed.tool_input,
                    "observation": observation,
                })

                # Append LLM output and Observation to messages
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}",
                })

        # Max steps reached
        logger.warning(f"Reached max steps {MAX_REACT_STEPS}, forcing stop")
        return "Sorry, I could not reach a satisfactory answer after multiple reasoning steps. Please try rephrasing your question."

    def _call_llm(self, messages: list[dict]) -> str:
        """Call LLM and get response"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=2048,
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
