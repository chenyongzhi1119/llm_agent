"""FastAPI application"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.routes import router
from src.tools.base import ToolRegistry
from src.tools.calculator import CalculatorTool
from src.tools.web_search import WebSearchTool
from src.tools.code_executor import CodeExecutorTool
from src.tools.weather import WeatherTool
from src.tools.file_rw import FileReadTool, FileWriteTool
from src.tools.youtube_summary import YouTubeSummaryTool
from src.tools.study_tutor import StudyTutorTool
from src.memory.long_term import init_db


def create_app() -> FastAPI:
    app = FastAPI(title="Multi-Tool ReAct Agent", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup():
        init_db()

        registry = ToolRegistry()
        registry.register(CalculatorTool())
        registry.register(WebSearchTool())
        registry.register(CodeExecutorTool())
        registry.register(WeatherTool())
        registry.register(FileReadTool())
        registry.register(FileWriteTool())
        registry.register(YouTubeSummaryTool())
        registry.register(StudyTutorTool())

        app.state.registry = registry
        logger.info(f"Registered tools: {[t.name for t in registry.list_tools()]}")

    app.include_router(router)
    return app


app = create_app()
