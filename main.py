"""Entry point - verify ReAct Agent basic flow"""

import sys
from loguru import logger

from src.tools.base import ToolRegistry
from src.tools.calculator import CalculatorTool
from src.agent.react import ReActAgent
from src.agent.graph import run_agent_graph

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
logger.add("logs/agent.log", level="DEBUG", rotation="1 MB")


def create_tool_registry() -> ToolRegistry:
    """Create and register all tools"""
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    return registry


def main():
    registry = create_tool_registry()

    print("=" * 50)
    print("Multi-Tool ReAct Agent")
    print("Type 'quit' to exit, 'graph' to toggle LangGraph mode")
    print("=" * 50)

    use_graph = False

    while True:
        question = input("\nYou: ").strip()
        if not question:
            continue
        if question.lower() == "quit":
            print("Bye!")
            break
        if question.lower() == "graph":
            use_graph = not use_graph
            mode = "LangGraph" if use_graph else "Hand-written ReAct"
            print(f"Switched to {mode} mode")
            continue

        try:
            if use_graph:
                answer = run_agent_graph(question, registry)
            else:
                agent = ReActAgent(registry)
                answer = agent.run(question)

            print(f"\nAgent: {answer}")
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
