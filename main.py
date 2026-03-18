"""Entry point - Multi-Tool ReAct Agent"""

import sys
from loguru import logger

from src.tools.base import ToolRegistry
from src.tools.calculator import CalculatorTool
from src.tools.web_search import WebSearchTool
from src.tools.code_executor import CodeExecutorTool
from src.tools.weather import WeatherTool
from src.tools.file_rw import FileReadTool, FileWriteTool
from src.tools.youtube_summary import YouTubeSummaryTool
from src.tools.study_tutor import StudyTutorTool
from src.agent.react import ReActAgent
from src.agent.graph import run_agent_graph

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
logger.add("logs/agent.log", level="DEBUG", rotation="1 MB")


def create_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(WebSearchTool())
    registry.register(CodeExecutorTool())
    registry.register(WeatherTool())
    registry.register(FileReadTool())
    registry.register(FileWriteTool())
    registry.register(YouTubeSummaryTool())
    registry.register(StudyTutorTool())
    return registry


def main():
    registry = create_tool_registry()

    print("=" * 50)
    print("Multi-Tool ReAct Agent")
    print(f"Tools: {[t.name for t in registry.list_tools()]}")
    print("Commands: 'quit' exit | 'graph' toggle LangGraph | 'clear' clear history | 'session <id>' switch session")
    print("=" * 50)

    use_graph = False
    session_id = "default"
    agent = ReActAgent(registry, session_id=session_id)
    print(f"Session: {agent.session_id}")

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

        if question.lower() == "clear":
            agent.clear_history()
            print("History cleared.")
            continue

        if question.lower().startswith("session "):
            new_session = question[8:].strip()
            if new_session:
                session_id = new_session
                agent = ReActAgent(registry, session_id=session_id)
                print(f"Switched to session: {agent.session_id}")
            continue

        try:
            if use_graph:
                answer = run_agent_graph(question, registry)
            else:
                answer = agent.run(question)

            print(f"\nAgent: {answer}")
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
