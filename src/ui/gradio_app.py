"""Gradio Web UI"""

import json
import httpx
import gradio as gr

API_BASE = "http://localhost:8000"

# Event type → display prefix
EVENT_PREFIX = {
    "thought":     "💭 **Thought:**",
    "action":      "🔧 **Action:**",
    "observation": "📋 **Observation:**",
}


def get_tools() -> list[str]:
    try:
        r = httpx.get(f"{API_BASE}/api/tools", timeout=5)
        return [f"- **{t['name']}**: {t['description']}" for t in r.json()["tools"]]
    except Exception:
        return ["(API not available — start server.py first)"]


def load_history(session_id: str) -> list[dict]:
    try:
        r = httpx.get(f"{API_BASE}/api/history", params={"session_id": session_id}, timeout=5)
        return [{"role": m["role"], "content": m["content"]} for m in r.json().get("messages", [])]
    except Exception:
        return []


def chat(question: str, history: list[dict], session_id: str):
    """
    Stream agent response.
    Yields: (history, reasoning_markdown, status_text)
    - reasoning_markdown updates in real-time as each SSE event arrives
    - status shows a spinner while thinking, clears on done, shows error in red
    """
    if not question.strip():
        yield history, "", ""
        return

    history = history + [{"role": "user", "content": question}]
    yield history, "", "⏳ Thinking..."

    reasoning_lines: list[str] = []
    final_answer = ""
    is_error = False

    try:
        with httpx.stream(
            "POST",
            f"{API_BASE}/api/chat",
            json={"question": question, "session_id": session_id},
            timeout=120,
        ) as response:
            if response.status_code != 200:
                raise RuntimeError(f"Server returned HTTP {response.status_code}")

            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw:
                    continue

                event = json.loads(raw)
                etype = event["type"]
                content = event["content"]

                if etype in EVENT_PREFIX:
                    # Append new reasoning step and yield immediately for real-time update
                    reasoning_lines.append(f"{EVENT_PREFIX[etype]} {content}")
                    yield history, "\n\n".join(reasoning_lines), "⏳ Reasoning..."

                elif etype == "answer":
                    final_answer = content

                elif etype == "error":
                    final_answer = content
                    is_error = True

    except Exception as e:
        final_answer = str(e)
        is_error = True

    # Format answer: red box for errors, normal for success
    if is_error:
        display = f"> ❌ **Error:** {final_answer}\n\n*Check that server.py is running and LM Studio is active.*"
        status = "❌ Error"
    else:
        display = final_answer or "(no response)"
        status = ""

    history = history + [{"role": "assistant", "content": display}]
    reasoning_done = "\n\n".join(reasoning_lines) if reasoning_lines else "*(no tool calls — direct answer)*"
    yield history, reasoning_done, status


def clear_history(session_id: str):
    try:
        httpx.delete(f"{API_BASE}/api/history", params={"session_id": session_id}, timeout=5)
        msg = f"History cleared for session: **{session_id}**"
    except Exception as e:
        msg = f"Could not clear history: {e}"
    return [], msg, ""


def build_ui() -> gr.Blocks:
    tools_list = get_tools()

    with gr.Blocks(title="Multi-Tool ReAct Agent") as demo:
        gr.Markdown("# Multi-Tool ReAct Agent")
        gr.Markdown("Powered by **Qwen** (LM Studio) · **LangGraph Multi-Agent** · Hand-written ReAct")

        with gr.Row():
            # ── Left: chat ────────────────────────────────────────────────
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=520,
                    buttons=["copy"],
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask anything...",
                        show_label=False,
                        scale=5,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                status_bar = gr.Markdown(value="")

                with gr.Row():
                    session_input = gr.Textbox(value="default", label="Session ID", scale=3)
                    clear_btn = gr.Button("Clear History", variant="stop", scale=1)

            # ── Right: reasoning chain + tools ────────────────────────────
            with gr.Column(scale=2):
                gr.Markdown("### Reasoning Chain")
                reasoning_box = gr.Markdown(value="*(reasoning steps appear here in real time)*")
                gr.Markdown("---")
                gr.Markdown("### Available Tools")
                gr.Markdown("\n".join(tools_list))

        gr.Examples(
            examples=[
                ["What is (128 * 37) + 512?"],
                ["What is the weather in London?"],
                ["Run this Python: print([i**2 for i in range(1, 6)])"],
                ["Search the web: what is LangGraph multi-agent?"],
                ["Explain neural networks at beginner level"],
                ["Save to notes.txt: meeting tomorrow 3pm"],
            ],
            inputs=msg_input,
        )

        # ── Event wiring ──────────────────────────────────────────────────
        def on_send(q, h, sid):
            yield from chat(q, h, sid)

        send_btn.click(
            on_send,
            inputs=[msg_input, chatbot, session_input],
            outputs=[chatbot, reasoning_box, status_bar],
        ).then(lambda: "", outputs=msg_input)

        msg_input.submit(
            on_send,
            inputs=[msg_input, chatbot, session_input],
            outputs=[chatbot, reasoning_box, status_bar],
        ).then(lambda: "", outputs=msg_input)

        clear_btn.click(
            clear_history,
            inputs=[session_input],
            outputs=[chatbot, reasoning_box, status_bar],
        )

        session_input.change(
            load_history,
            inputs=[session_input],
            outputs=[chatbot],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
