# Multi-Tool ReAct Agent

A production-ready AI agent built with hand-written ReAct loop + LangGraph, powered by local Qwen model via LM Studio.

## Architecture

```
┌───────────────────────────────────────────────┐
│             Gradio Web UI (:7860)             │
├───────────────────────────────────────────────┤
│            FastAPI Backend (:8000)            │
│            SSE streaming responses            │
├───────────────────────────────────────────────┤
│         ReAct Agent Core (LangGraph)          │
│  ┌────────────┐ ┌────────────┐ ┌───────────┐  │
│  │Hand-written│ │  LangGraph │ │   Memory  │  │
│  │  ReAct loop│ │ state graph│ │ short+long│  │
│  └────────────┘ └────────────┘ └───────────┘  │
├───────────────────────────────────────────────┤
│                  Tool Layer                   │
│  calculator    web_search    code_executor    │
│  weather       file_rw       youtube_summary  │
│  study_tutor                                  │
├───────────────────────────────────────────────┤
│        Local LLM (Qwen via LM Studio)         │
└───────────────────────────────────────────────┘
```

## Quick Start

**Prerequisites:** LM Studio running with Qwen model loaded on `localhost:1234`

```powershell
# Terminal 1 - Backend
& "C:\Program Files\Python311\python.exe" server.py

# Terminal 2 - UI
& "C:\Program Files\Python311\python.exe" src/ui/gradio_app.py
```

Open `http://localhost:7860` in your browser.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Qwen (LM Studio local, OpenAI-compatible API) |
| Agent | Hand-written ReAct + LangGraph state graph |
| Backend | FastAPI + SSE streaming |
| Frontend | Gradio |
| Memory | Short-term (sliding window) + Long-term (SQLite) |
| Tools | yt-dlp, DuckDuckGo, subprocess sandbox, wttr.in, httpx |

## Tools

| Tool | Description |
|------|-------------|
| `calculator` | Evaluate math expressions safely |
| `web_search` | DuckDuckGo search, no API key needed |
| `code_executor` | Sandboxed Python execution with timeout |
| `weather` | Real-time weather via wttr.in |
| `file_read` | Read files from data/ directory |
| `file_write` | Write files to data/ directory |
| `youtube_summary` | Extract subtitles and summarize YouTube videos |
| `study_tutor` | Explain topics, generate quizzes, grade answers |

## Configuration

Edit `config/settings.py`:

```python
LLM_BASE_URL = "http://localhost:1234/v1"  # LM Studio URL
LLM_MODEL    = "qwen2.5"                   # model name as shown in LM Studio
MAX_REACT_STEPS     = 10                   # max reasoning iterations
MAX_CONTEXT_MESSAGES = 20                  # sliding window size
```

## Extending Tools

```python
from src.tools.base import Tool

class MyTool(Tool):
    name = "my_tool"
    description = "One-line description for the LLM"
    parameters = {
        "query": {"description": "input text", "required": True},
    }

    def execute(self, **kwargs) -> str:
        return f"result for {kwargs['query']}"
```

Register in `server.py` / `main.py`:
```python
registry.register(MyTool())
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat` | SSE streaming chat |
| GET | `/api/tools` | List registered tools |
| GET | `/api/history?session_id=x` | Get conversation history |
| DELETE | `/api/history?session_id=x` | Clear history |
| GET | `/api/sessions` | List all sessions |
| GET | `/docs` | Swagger UI |

---

## Optimizations

### 1. System Prompt Token Compression (−42%)

The initial system prompt + 8 tool descriptions totalled ~703 tokens. On CPU-only inference every token adds latency, and LM Studio logs showed 2392 tokens per request.

Rewrote all tool descriptions to be maximally concise:

```python
# Before — 703 tokens total
description = "Search the web using DuckDuckGo. Returns top results with title, URL, and snippet."
parameters  = {"query": {"description": "The search query string", "required": True}, ...}

# After — 407 tokens total (−42%)
description = "Search the web via DuckDuckGo. Returns titles, URLs, snippets."
parameters  = {"query": {"description": "search query", "required": True}, ...}
```

Also compressed the ReAct format instructions from a verbose paragraph to a minimal template.

**Result:** avg latency dropped from ~4s to ~1.85s.

### 2. Forced Tool Invocation via Prompt Rules

Qwen would bypass tools by fabricating Observation lines (e.g. inventing weather data or calculating mentally). Added explicit STRICT RULES to the system prompt:

```
NEVER write an Observation line yourself.
Observations are ONLY filled by the system after a real tool call.
NEVER guess or make up numbers, weather, code output, or file content.
```

**Result:** tool call rate 0% → 100%.

### 3. Parser-Level Hallucination Defense

Added a second layer of protection in `parser.py` that strips any self-written Observation before extracting the Action, forcing the agent back into the correct tool-call path regardless of prompt compliance:

```python
def _strip_fabricated_observation(text: str) -> str:
    return re.split(r"\nObservation\s*:", text)[0].strip()
```

---

## Technical Challenges

### Challenge 1 — LLM Hallucinating Tool Results

**Problem:** Qwen in ReAct format fabricated `Observation` lines instead of calling tools. Given "What is the weather in Tokyo?", it would write `Observation: The weather is 22°C sunny` without invoking the weather tool.

**Root cause:** During training the model learned the ReAct pattern (Action → Observation), so it tries to "complete the sequence" by generating a plausible Observation.

**Solution:** Two-layer fix — strict prompt rules (see above) + parser-level stripping. Verified by checking that SSE responses contain an `action` event before any `observation` event.

---

### Challenge 2 — Stale Process Holding Port After Code Update

**Problem:** After updating `prompt.py`, the running FastAPI server kept using the old in-memory code. Restarting appeared to work (new process started) but the old process still held port 8000, so the new process silently exited. The server continued serving wrong answers from stale code.

**Diagnosis:** LM Studio debug logs showed requests still processing 2392 tokens even after the prompt was compressed to ~407 tokens — clear evidence of stale code.

**Solution:**
```powershell
netstat -ano | findstr :8000   # find PID
taskkill /F /PID <pid>         # hard-kill old process, then restart
```

---

### Challenge 3 — Third-Party API Breaking Changes

Two dependencies changed APIs silently:

| Library | Change | Fix |
|---------|--------|-----|
| `duckduckgo_search` | Renamed to `ddgs` | Graceful fallback import: `try ddgs except ImportError use duckduckgo_search` |
| `wttr.in` | Response body moved from root to nested `"data"` key | `data = r.json().get("data", r.json())` |

---

### Challenge 4 — Gradio 6.x API Breaking Changes

Three simultaneous breaking changes when upgrading to Gradio 6:

| Parameter | Old (v5) | New (v6) |
|-----------|----------|----------|
| `theme` | `gr.Blocks(theme=...)` | `launch(theme=...)` |
| Copy button | `Chatbot(show_copy_button=True)` | `Chatbot(buttons=["copy"])` |
| Message type | `Chatbot(type="messages")` | parameter removed; dict format is default |
| History format | `[[user, assistant], ...]` | `[{"role": "user", "content": ...}, ...]` |

---

### Challenge 5 — Python Multi-Installation Conflict (Windows)

**Problem:** `python` resolved to msys64 Python 3.12 (no pip), but all dependencies were installed to `C:\Program Files\Python311`. Running `python main.py` raised `ModuleNotFoundError`.

**Solution:** Always use the explicit path `"C:\Program Files\Python311\python.exe"` in all commands.

---

## Performance

Measured on CPU-only machine (no GPU), Qwen model via LM Studio:

| Metric | Value |
|--------|-------|
| Direct answer (no tool) | ~1.2s avg |
| Tool-call round-trip | ~2.1s avg |
| Overall average | **1.85s** |
| Min | 0.45s |
| Max | 2.92s |
| UX rating | **Excellent (< 3s)** |

**Further speed improvement:** switch to Q4_K_M quantization in LM Studio — typically 3–5× faster on CPU with minimal quality loss.

## Project Structure

```
llm_agent/
├── config/settings.py        # LM Studio URL, model name, limits
├── src/
│   ├── agent/
│   │   ├── react.py          # Hand-written ReAct loop
│   │   ├── graph.py          # LangGraph state graph
│   │   ├── prompt.py         # Optimized ReAct system prompt
│   │   └── parser.py         # Output parser + hallucination defense
│   ├── tools/
│   │   ├── base.py           # Tool ABC + ToolRegistry
│   │   ├── calculator.py
│   │   ├── web_search.py
│   │   ├── code_executor.py  # Sandboxed subprocess execution
│   │   ├── weather.py
│   │   ├── file_rw.py
│   │   ├── youtube_summary.py
│   │   └── study_tutor.py
│   ├── memory/
│   │   ├── short_term.py     # Sliding window context manager
│   │   └── long_term.py      # SQLite persistence
│   ├── api/
│   │   ├── app.py            # FastAPI app factory
│   │   ├── routes.py         # Endpoints + SSE streaming
│   │   └── schemas.py        # Pydantic models
│   └── ui/
│       └── gradio_app.py     # Gradio chat interface
├── tests/
│   └── test_performance.py   # Latency benchmarks
├── main.py                   # CLI entry point
└── server.py                 # FastAPI server entry point
```

## License

MIT
