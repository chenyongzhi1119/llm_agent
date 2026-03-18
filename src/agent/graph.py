"""
Multi-Agent LangGraph implementation with memory.

Architecture:
  Supervisor Agent
  ├── MathAgent       (calculator)
  ├── WebAgent        (web_search)
  ├── CodeAgent       (code_executor)
  ├── WeatherAgent    (weather)
  ├── FileAgent       (file_read, file_write)
  └── KnowledgeAgent  (study_tutor, youtube_summary)

The Supervisor routes each user question to the most appropriate sub-agent.
Each sub-agent runs its own mini ReAct loop with only the tools it owns.
"""

import uuid
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from openai import OpenAI
from loguru import logger

from src.agent.parser import parse_react_output, ReActAction, ReActFinish
from src.agent.prompt import REACT_USER_PROMPT
from src.tools.base import ToolRegistry
from src.memory.short_term import ShortTermMemory
from src.memory.long_term import init_db, save_message, load_history
from config.settings import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, MAX_REACT_STEPS


# ── LLM client ────────────────────────────────────────────────────────────────

_client: OpenAI | None = None

def _llm() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    return _client

def _call(messages: list[dict]) -> str:
    resp = _llm().chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=512,
    )
    return resp.choices[0].message.content or ""


# ── Sub-agent tool groups ─────────────────────────────────────────────────────

SUB_AGENT_TOOLS: dict[str, list[str]] = {
    "MathAgent":      ["calculator"],
    "WebAgent":       ["web_search"],
    "CodeAgent":      ["code_executor"],
    "WeatherAgent":   ["weather"],
    "FileAgent":      ["file_read", "file_write"],
    "KnowledgeAgent": ["study_tutor", "youtube_summary"],
}


# ── Keyword-based supervisor router (zero LLM calls) ─────────────────────────

import re as _re

_ROUTING_RULES: list[tuple[str, list[str]]] = [
    ("MathAgent",      [r"\bcalculate\b", r"\bmath\b", r"\bcompute\b", r"\d+\s*[\+\-\*/]\s*\d+",
                        r"\bwhat is\s+[\d\s\+\-\*\/\(\)]+", r"\bresult of\b", r"\bsum\b", r"\bproduct\b"]),
    ("WeatherAgent",   [r"\bweather\b", r"\btemperature\b", r"\braining\b", r"\bforecast\b", r"\bhumidity\b"]),
    ("CodeAgent",      [r"\brun\b.*python", r"\bexecute\b.*code", r"\bcode\b", r"\bscript\b", r"\bprint\s*\("]),
    ("FileAgent",      [r"\bsave\b.*\.txt", r"\bwrite\b.*file", r"\bread\b.*file", r"\bfile\b"]),
    ("KnowledgeAgent", [r"\bexplain\b", r"\bwhat is\b", r"\bhow does\b", r"\byoutube\b",
                        r"\bquiz\b", r"\blearn\b", r"\bstudy\b"]),
    ("WebAgent",       [r"\bsearch\b", r"\blook up\b", r"\bfind\b", r"\bnews\b", r"\blatest\b"]),
]

def _keyword_route(question: str) -> str:
    """Route to sub-agent using keyword matching. O(1) cost, no LLM call."""
    q = question.lower()
    for agent, patterns in _ROUTING_RULES:
        if any(_re.search(p, q) for p in patterns):
            logger.info(f"[Supervisor] keyword route -> {agent}")
            return agent
    logger.info("[Supervisor] no keyword match, defaulting to WebAgent")
    return "WebAgent"

SUB_AGENT_PROMPT = """You are a specialist agent. Be concise. Use tools, never fabricate.

RULES:
- NEVER write Observation yourself. System fills it after tool call.
- NEVER guess results. Call a tool.
- Keep Final Answer short and direct.

Format:
Thought: reason
Action: tool_name
Action Input: {{"param": "value"}}
Thought: done
Final Answer: answer

Tools:
{tools_description}

Answer in English."""


# ── Graph state ────────────────────────────────────────────────────────────────

class MultiAgentState(TypedDict):
    question: str
    session_id: str
    messages: list[dict]        # full conversation (with memory)
    assigned_agent: str         # which sub-agent was chosen
    final_answer: str
    tool_registry: ToolRegistry
    current_step: int
    consecutive_errors: int


# ── Nodes ──────────────────────────────────────────────────────────────────────

def supervisor_node(state: MultiAgentState) -> MultiAgentState:
    """Route to sub-agent using fast keyword matching (no LLM call)."""
    state["assigned_agent"] = _keyword_route(state["question"])
    return state


def sub_agent_node(state: MultiAgentState) -> MultiAgentState:
    """Run the assigned sub-agent's ReAct loop."""
    agent_name = state["assigned_agent"]
    tool_names = SUB_AGENT_TOOLS.get(agent_name, [])
    registry = state["tool_registry"]

    # Build a mini registry with only this agent's tools
    mini_registry = ToolRegistry()
    for name in tool_names:
        tool = registry.get(name)
        if tool:
            mini_registry.register(tool)

    # Build messages: memory context + new question
    messages = list(state["messages"])  # already has system prompt + history
    messages.append({"role": "user", "content": REACT_USER_PROMPT.format(question=state["question"])})

    # Replace system prompt with sub-agent prompt
    sub_system = SUB_AGENT_PROMPT.format(tools_description=mini_registry.get_tools_prompt())
    if messages and messages[0]["role"] == "system":
        messages[0] = {"role": "system", "content": sub_system}
    else:
        messages.insert(0, {"role": "system", "content": sub_system})

    consecutive_errors = 0

    for step in range(MAX_REACT_STEPS):
        logger.info(f"[{agent_name}] step {step + 1}")
        response = _call(messages)
        logger.info(f"[{agent_name}] output:\n{response}")

        parsed = parse_react_output(response)

        if isinstance(parsed, ReActFinish):
            logger.info(f"[{agent_name}] final answer: {parsed.answer}")
            state["final_answer"] = parsed.answer
            return state

        if isinstance(parsed, ReActAction):
            tool = mini_registry.get(parsed.tool_name)
            if tool is None:
                observation = f"Error: tool '{parsed.tool_name}' not found in {agent_name}. Available: {[t.name for t in mini_registry.list_tools()]}"
            else:
                try:
                    observation = tool.execute(**parsed.tool_input)
                except Exception as e:
                    observation = f"Error: {e}"

            logger.info(f"[{agent_name}] observation: {observation}")

            if observation.startswith("Error:"):
                consecutive_errors += 1
                if consecutive_errors >= 3:
                    state["final_answer"] = f"Unable to complete: {observation}"
                    return state
            else:
                consecutive_errors = 0

            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Observation: {observation}"})

    state["final_answer"] = "Could not reach an answer within the step limit."
    return state


# ── Graph ──────────────────────────────────────────────────────────────────────

def build_multi_agent_graph():
    graph = StateGraph(MultiAgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("sub_agent", sub_agent_node)
    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "sub_agent")
    graph.add_edge("sub_agent", END)
    return graph.compile()


_graph = None

def run_agent_graph(question: str, tool_registry: ToolRegistry, session_id: str = "default") -> str:
    """
    Run the multi-agent graph with memory.

    Args:
        question: user question
        tool_registry: all registered tools
        session_id: conversation session for memory persistence

    Returns:
        final answer string
    """
    global _graph
    if _graph is None:
        _graph = build_multi_agent_graph()

    init_db()

    # Build short-term memory with history
    short_term = ShortTermMemory()
    short_term.set_system_prompt("")  # placeholder; sub_agent_node overrides it
    for msg in load_history(session_id):
        short_term.add(msg["role"], msg["content"])

    save_message(session_id, "user", question)

    initial_state: MultiAgentState = {
        "question": question,
        "session_id": session_id,
        "messages": short_term.get_messages(),
        "assigned_agent": "",
        "final_answer": "",
        "tool_registry": tool_registry,
        "current_step": 0,
        "consecutive_errors": 0,
    }

    final_state = _graph.invoke(initial_state)
    answer = final_state.get("final_answer", "No answer returned.")
    save_message(session_id, "assistant", answer)
    return answer
