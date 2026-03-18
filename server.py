"""Start the FastAPI server"""

import sys
from loguru import logger
import uvicorn
from config.settings import API_HOST, API_PORT

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
logger.add("logs/agent.log", level="DEBUG", rotation="1 MB")

if __name__ == "__main__":
    logger.info(f"Starting server at http://{API_HOST}:{API_PORT}")
    logger.info(f"API docs: http://localhost:{API_PORT}/docs")
    uvicorn.run("src.api.app:app", host=API_HOST, port=API_PORT, reload=False)
