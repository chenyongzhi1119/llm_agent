"""API routes"""

import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.api.schemas import ChatRequest, ToolInfo, MessageItem
from src.tools.base import ToolRegistry
from src.memory.long_term import load_history, clear_history as db_clear_history

router = APIRouter()

# Registry is injected via app state
def get_registry(request) -> ToolRegistry:
    return request.app.state.registry


# ---- SSE streaming agent ----

def _make_event(event_type: str, content: str) -> str:
    data = json.dumps({"type": event_type, "content": content})
    return f"data: {data}\n\n"


async def _stream_agent(question: str, session_id: str, registry: ToolRegistry):
    """Run ReAct agent and stream each step as SSE events"""
    from openai import OpenAI
    from src.agent.prompt import REACT_SYSTEM_PROMPT, REACT_USER_PROMPT
    from src.agent.parser import parse_react_output, ReActAction, ReActFinish
    from src.agent.react import ReActAgent
    from src.memory.long_term import save_message, init_db
    from config.settings import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, MAX_REACT_STEPS
    from src.memory.short_term import ShortTermMemory
    from loguru import logger

    try:
        init_db()
        client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

        # Build short-term memory with history
        short_term = ShortTermMemory()
        system_prompt = REACT_SYSTEM_PROMPT.format(
            tools_description=registry.get_tools_prompt()
        )
        short_term.set_system_prompt(system_prompt)
        for msg in load_history(session_id):
            short_term.add(msg["role"], msg["content"])

        short_term.add("user", REACT_USER_PROMPT.format(question=question))
        save_message(session_id, "user", question)

        for step in range(MAX_REACT_STEPS):
            # Call LLM
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=short_term.get_messages(),
                temperature=0.1,
                max_tokens=2048,
            )
            llm_output = response.choices[0].message.content or ""

            # Parse
            parsed = parse_react_output(llm_output)

            # Extract and stream Thought
            import re
            thought_match = re.search(r"Thought\s*:\s*(.+?)(?=\nAction|\nFinal Answer|$)", llm_output, re.DOTALL)
            if thought_match:
                yield _make_event("thought", thought_match.group(1).strip())

            if isinstance(parsed, ReActFinish):
                save_message(session_id, "assistant", parsed.answer)
                short_term.add("assistant", parsed.answer)
                yield _make_event("answer", parsed.answer)
                return

            if isinstance(parsed, ReActAction):
                yield _make_event("action", f"{parsed.tool_name}({json.dumps(parsed.tool_input)})")

                tool = registry.get(parsed.tool_name)
                if tool is None:
                    obs = f"Error: tool '{parsed.tool_name}' not found"
                else:
                    try:
                        obs = tool.execute(**parsed.tool_input)
                    except Exception as e:
                        obs = f"Tool error: {e}"

                yield _make_event("observation", obs)
                short_term.add("assistant", llm_output)
                short_term.add("user", f"Observation: {obs}")

        # Max steps
        answer = "Sorry, could not reach an answer after maximum reasoning steps."
        save_message(session_id, "assistant", answer)
        yield _make_event("answer", answer)

    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield _make_event("error", str(e))


# ---- Endpoints ----

from fastapi import Request

@router.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    registry = get_registry(request)
    return StreamingResponse(
        _stream_agent(req.question, req.session_id, registry),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/api/history")
async def get_history(session_id: str = "default", limit: int = 50):
    messages = load_history(session_id, limit=limit)
    return {"session_id": session_id, "messages": messages}


@router.delete("/api/history")
async def delete_history(session_id: str = "default"):
    db_clear_history(session_id)
    return {"status": "ok", "session_id": session_id}


@router.get("/api/tools")
async def list_tools(request: Request):
    registry = get_registry(request)
    tools = [
        ToolInfo(name=t.name, description=t.description)
        for t in registry.list_tools()
    ]
    return {"tools": tools}


@router.get("/api/sessions")
async def list_sessions():
    from src.memory.long_term import list_sessions
    return {"sessions": list_sessions()}
