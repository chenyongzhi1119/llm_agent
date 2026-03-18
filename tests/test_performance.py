"""Performance and latency tests for the ReAct Agent"""

import time
import json
import statistics
import httpx

API_BASE = "http://localhost:8000"
SESSION = "perf-test"

# ── helpers ──────────────────────────────────────────────────────────────────

def call_chat(question: str, timeout: int = 120) -> dict:
    """Call /api/chat and collect timing + SSE events"""
    events = []
    t0 = time.perf_counter()
    first_token_time = None

    with httpx.stream(
        "POST",
        f"{API_BASE}/api/chat",
        json={"question": question, "session_id": SESSION},
        timeout=timeout,
    ) as resp:
        for line in resp.iter_lines():
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if not raw:
                continue
            event = json.loads(raw)
            ts = time.perf_counter()
            if first_token_time is None:
                first_token_time = ts - t0
            events.append({"type": event["type"], "content": event["content"], "ts": ts - t0})

    total = time.perf_counter() - t0
    answer = next((e["content"] for e in reversed(events) if e["type"] == "answer"), "")
    steps = [e for e in events if e["type"] in ("thought", "action", "observation")]

    return {
        "question": question,
        "total_s": round(total, 2),
        "first_event_s": round(first_token_time or total, 2),
        "steps": len(steps),
        "tool_calls": len([e for e in events if e["type"] == "action"]),
        "answer": answer[:80] + ("..." if len(answer) > 80 else ""),
        "events": events,
    }


def print_result(r: dict, idx: int):
    tag = "TOOL" if r["tool_calls"] > 0 else "DIRECT"
    print(f"\n  [{idx}] [{tag}] {r['question'][:60]}")
    print(f"       Total: {r['total_s']}s  |  First event: {r['first_event_s']}s  |  Tool calls: {r['tool_calls']}")
    print(f"       Answer: {r['answer']}")


# ── test cases ────────────────────────────────────────────────────────────────

CASES = [
    # Direct answer (no tool needed)
    ("direct", "What is the capital of France?"),
    ("direct", "What does CPU stand for?"),
    # Calculator
    ("tool", "What is 128 * 37 + 512?"),
    ("tool", "Calculate (99 + 1) / 4"),
    # Weather
    ("tool", "What is the weather in Tokyo?"),
    # Code executor
    ("tool", "Run Python: print(sum(range(10)))"),
    # File write
    ("tool", "Save to test_perf.txt: hello world"),
]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("ReAct Agent Performance Test")
    print("=" * 60)

    # Clean session
    httpx.delete(f"{API_BASE}/api/history", params={"session_id": SESSION}, timeout=5)

    results = []
    for i, (category, question) in enumerate(CASES, 1):
        print(f"\nRunning [{i}/{len(CASES)}]: {question[:55]}...")
        try:
            r = call_chat(question)
            r["category"] = category
            results.append(r)
            print_result(r, i)
        except Exception as e:
            print(f"  ERROR: {e}")

    # ── summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_times = [r["total_s"] for r in results]
    direct = [r["total_s"] for r in results if r["category"] == "direct"]
    tool   = [r["total_s"] for r in results if r["category"] == "tool"]

    print(f"\nTotal cases:       {len(results)}")
    print(f"Direct (no tool):  {len(direct)} cases")
    print(f"Tool-call:         {len(tool)} cases")

    if all_times:
        print(f"\nOverall latency:")
        print(f"  Min:    {min(all_times):.2f}s")
        print(f"  Max:    {max(all_times):.2f}s")
        print(f"  Avg:    {statistics.mean(all_times):.2f}s")
        print(f"  Median: {statistics.median(all_times):.2f}s")

    if direct:
        print(f"\nDirect answer latency (avg): {statistics.mean(direct):.2f}s")
    if tool:
        print(f"Tool-call latency   (avg): {statistics.mean(tool):.2f}s")

    # UX rating
    avg = statistics.mean(all_times) if all_times else 99
    print(f"\nUser experience rating:")
    if avg < 3:
        rating = "Excellent (< 3s)"
    elif avg < 6:
        rating = "Good (3-6s)"
    elif avg < 10:
        rating = "Acceptable (6-10s) - consider smaller model"
    else:
        rating = "Poor (> 10s) - needs optimization"
    print(f"  {rating}")

    # Bottleneck hints
    print(f"\nBottleneck analysis:")
    if direct and statistics.mean(direct) > 5:
        print("  - LLM inference is slow (pure CPU). Try: smaller quantization (Q4_K_M)")
    if tool and statistics.mean(tool) > 8:
        print("  - Tool-call round-trips add overhead. Consider streaming tool results.")
    if avg < 10:
        print("  - System prompt size: ~407 tokens (already optimized)")

    # Clean up
    httpx.delete(f"{API_BASE}/api/history", params={"session_id": SESSION}, timeout=5)
    print("\nTest complete.")


if __name__ == "__main__":
    main()
