# Multi-Tool ReAct Agent

A multi-tool intelligent Agent built with **LangGraph + hand-written ReAct loop**, powered by local Qwen model via LM Studio. Supports tool invocation, multi-turn conversation, and provides both CLI and Web UI interfaces.

## Architecture

```
┌─────────────────────────────────────────────┐
│                Web UI (Gradio)              │
├─────────────────────────────────────────────┤
│              FastAPI Backend                │
├─────────────────────────────────────────────┤
│         ReAct Agent Core (LangGraph)        │
│  ┌──────────┐ ┌───────────┐ ┌───────────┐  │
│  │ Reasoning │ │Tool Router│ │  Memory   │  │
│  └──────────┘ └───────────┘ └───────────┘  │
├─────────────────────────────────────────────┤
│               Tools Layer                   │
│  YouTube Summary · Study Tutor · Web Search │
│  Code Executor  · Weather     · File R/W   │
├─────────────────────────────────────────────┤
│          Local LLM (Qwen via LM Studio)     │
└─────────────────────────────────────────────┘
```

## Features

- **Dual ReAct Implementation** — both a hand-written ReAct loop and a LangGraph state graph version, switchable at runtime
- **Dynamic Tool Registry** — unified tool interface with auto-registration; easily extensible
- **Local LLM** — runs Qwen2.5 locally via LM Studio (OpenAI-compatible API), no cloud dependency
- **Streaming Output** — SSE-based real-time streaming of the agent's reasoning chain (Thought → Action → Observation)
- **Multi-turn Memory** — short-term sliding window + long-term SQLite persistence

## Tech Stack

| Component | Choice |
|-----------|--------|
| LLM | Qwen2.5 (local, via LM Studio) |
| Agent Framework | LangGraph + hand-written ReAct |
| Backend | FastAPI + SSE |
| Frontend | Gradio |
| Tool Management | Custom Tool Registry |
| Storage | SQLite |

## Prerequisites

- **Python** >= 3.11
- **LM Studio** — download from [lmstudio.ai](https://lmstudio.ai), load a Qwen2.5 model, and start the local server (default: `http://localhost:1234/v1`)

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/chenyongzhi1119/llm_agent.git
cd llm_agent
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start LM Studio

1. Open LM Studio and download a Qwen2.5 model (e.g. `qwen2.5-7b-instruct`)
2. Go to the **Local Server** tab and click **Start Server**
3. The server runs at `http://localhost:1234/v1` by default

### 5. Run the agent

```bash
python main.py
```

Type your question in the CLI. Type `graph` to switch between hand-written ReAct and LangGraph mode, or `quit` to exit.

## Configuration

Edit `config/settings.py` to customize:

```python
LLM_BASE_URL = "http://localhost:1234/v1"  # LM Studio server URL
LLM_MODEL = "qwen2.5"                      # Model name
MAX_REACT_STEPS = 10                        # Max reasoning steps
MAX_CONTEXT_MESSAGES = 20                   # Conversation history window
```

## Project Structure

```
llm_agent/
├── main.py                 # Entry point (CLI)
├── config/
│   └── settings.py         # Configuration
├── src/
│   ├── agent/
│   │   ├── react.py        # Hand-written ReAct core logic
│   │   ├── graph.py        # LangGraph state graph
│   │   ├── prompt.py       # Prompt templates
│   │   └── parser.py       # Output parser
│   ├── tools/
│   │   ├── base.py         # Tool base class + Registry
│   │   └── calculator.py   # Calculator tool (example)
│   ├── memory/             # Short-term & long-term memory
│   ├── api/                # FastAPI backend
│   └── ui/                 # Gradio frontend
├── tests/
├── data/
└── logs/
```

## Extending Tools

Create a new tool by subclassing `Tool`:

```python
from src.tools.base import Tool

class MyTool(Tool):
    name = "my_tool"
    description = "Description of what this tool does"
    parameters = {
        "query": {"description": "Input query", "required": True},
    }

    def execute(self, **kwargs) -> str:
        query = kwargs["query"]
        return f"Result for: {query}"
```

Register it in `main.py`:

```python
from src.tools.my_tool import MyTool

registry.register(MyTool())
```

## License

MIT
