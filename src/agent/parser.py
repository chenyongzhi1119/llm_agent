"""ReAct output parser - extracts Action/Final Answer from LLM output"""

import json
import re
from dataclasses import dataclass


@dataclass
class ReActAction:
    """Parsed tool call"""
    tool_name: str
    tool_input: dict


@dataclass
class ReActFinish:
    """Parsed final answer"""
    answer: str


def parse_react_output(text: str) -> ReActAction | ReActFinish:
    """
    Parse LLM output to extract Action or Final Answer.

    Returns:
        ReActAction: if the LLM wants to call a tool
        ReActFinish: if the LLM gives a final answer
    """
    # Check Final Answer first
    final_match = re.search(r"Final Answer\s*:\s*(.+)", text, re.DOTALL)
    if final_match:
        return ReActFinish(answer=final_match.group(1).strip())

    # Extract Action and Action Input
    action_match = re.search(r"Action\s*:\s*(.+?)(?:\n|$)", text)
    input_match = re.search(r"Action Input\s*:\s*(.+?)(?:\n|$)", text, re.DOTALL)

    if not action_match:
        # No Action or Final Answer found, treat as final answer
        thought_match = re.search(r"Thought\s*:\s*(.+)", text, re.DOTALL)
        if thought_match:
            return ReActFinish(answer=thought_match.group(1).strip())
        return ReActFinish(answer=text.strip())

    tool_name = action_match.group(1).strip()

    # Parse Action Input
    tool_input = {}
    if input_match:
        raw_input = input_match.group(1).strip()
        try:
            tool_input = json.loads(raw_input)
        except json.JSONDecodeError:
            # Not valid JSON, pass as single string parameter
            tool_input = {"input": raw_input}

    return ReActAction(tool_name=tool_name, tool_input=tool_input)
