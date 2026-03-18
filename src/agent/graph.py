"""LangGraph state graph - implements ReAct loop as a graph"""

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from loguru import logger

from src.agent.parser import parse_react_output, ReActAction, ReActFinish
from src.agent.prompt import REACT_SYSTEM_PROMPT, REACT_USER_PROMPT
from src.tools.base import ToolRegistry
from config.settings import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, MAX_REACT_STEPS

from openai import OpenAI


# --- State definition ---

class AgentState(TypedDict):
    """Agent state passed between graph nodes"""
    messages: list[dict]          # Full message history
    current_step: int             # Current reasoning step count
    final_answer: str             # Final answer
    tool_registry: ToolRegistry   # Tool registry reference


# --- LLM client ---

_client = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    return _client


# --- Graph nodes ---

def reason_node(state: AgentState) -> AgentState:
    """Reasoning node: call LLM to think"""
    logger.info(f"--- reason_node (step {state['current_step'] + 1}) ---")

    client = _get_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=state["messages"],
        temperature=0.1,
        max_tokens=2048,
    )
    llm_output = response.choices[0].message.content or ""
    logger.info(f"LLM output:\n{llm_output}")

    # Append LLM reply to message history
    state["messages"].append({"role": "assistant", "content": llm_output})
    state["current_step"] += 1

    return state


def act_node(state: AgentState) -> AgentState:
    """Action node: parse LLM output and execute tool call"""
    last_message = state["messages"][-1]["content"]
    parsed = parse_react_output(last_message)

    if isinstance(parsed, ReActFinish):
        state["final_answer"] = parsed.answer
        return state

    if isinstance(parsed, ReActAction):
        logger.info(f"Calling tool: {parsed.tool_name}, input: {parsed.tool_input}")
        registry = state["tool_registry"]
        tool = registry.get(parsed.tool_name)

        if tool is None:
            observation = f"Error: tool '{parsed.tool_name}' not found. Available tools: {[t.name for t in registry.list_tools()]}"
        else:
            try:
                observation = tool.execute(**parsed.tool_input)
            except Exception as e:
                observation = f"Tool execution error: {e}"

        logger.info(f"Observation: {observation}")
        state["messages"].append({
            "role": "user",
            "content": f"Observation: {observation}",
        })

    return state


# --- Conditional routing ---

def should_continue(state: AgentState) -> str:
    """Decide whether to continue reasoning"""
    if state.get("final_answer"):
        return "end"

    if state["current_step"] >= MAX_REACT_STEPS:
        state["final_answer"] = "Sorry, I could not reach a satisfactory answer after multiple reasoning steps. Please try rephrasing your question."
        return "end"

    return "continue"


# --- Build graph ---

def build_agent_graph() -> StateGraph:
    """Build the LangGraph state graph"""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("reason", reason_node)
    graph.add_node("act", act_node)

    # Set entry point
    graph.set_entry_point("reason")

    # Add edges
    graph.add_edge("reason", "act")
    graph.add_conditional_edges(
        "act",
        should_continue,
        {
            "continue": "reason",
            "end": END,
        },
    )

    return graph.compile()


def run_agent_graph(question: str, tool_registry: ToolRegistry) -> str:
    """
    Run Agent via LangGraph.

    Args:
        question: User question
        tool_registry: Tool registry

    Returns:
        Final answer
    """
    graph = build_agent_graph()

    system_prompt = REACT_SYSTEM_PROMPT.format(
        tools_description=tool_registry.get_tools_prompt()
    )

    initial_state: AgentState = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": REACT_USER_PROMPT.format(question=question)},
        ],
        "current_step": 0,
        "final_answer": "",
        "tool_registry": tool_registry,
    }

    final_state = graph.invoke(initial_state)
    return final_state.get("final_answer", "Failed to get answer")
