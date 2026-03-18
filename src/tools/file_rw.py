"""File read/write tool - with path safety check"""

import os

from src.tools.base import Tool
from config.settings import ALLOWED_FILE_DIR


class FileReadTool(Tool):
    name = "file_read"
    description = "Read a file from the data/ directory."
    parameters = {
        "path": {"description": "filename e.g. 'notes.txt'", "required": True},
    }

    def execute(self, **kwargs) -> str:
        path = kwargs.get("path", "")
        if not path:
            return "Error: please provide a file path"

        full_path = os.path.normpath(os.path.join(ALLOWED_FILE_DIR, path))

        # Safety: ensure path stays within allowed directory
        allowed = os.path.normpath(os.path.abspath(ALLOWED_FILE_DIR))
        target = os.path.normpath(os.path.abspath(full_path))
        if not target.startswith(allowed):
            return f"Error: access denied. Only files under '{ALLOWED_FILE_DIR}' are allowed."

        if not os.path.exists(full_path):
            return f"Error: file not found: {path}"

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            if not content:
                return "(file is empty)"
            return content
        except Exception as e:
            return f"File read error: {e}"


class FileWriteTool(Tool):
    name = "file_write"
    description = "Write content to a file in the data/ directory."
    parameters = {
        "path": {"description": "filename e.g. 'notes.txt'", "required": True},
        "content": {"description": "text to write", "required": True},
    }

    def execute(self, **kwargs) -> str:
        path = kwargs.get("path", "")
        content = kwargs.get("content", "")
        if not path:
            return "Error: please provide a file path"

        full_path = os.path.normpath(os.path.join(ALLOWED_FILE_DIR, path))

        # Safety: ensure path stays within allowed directory
        allowed = os.path.normpath(os.path.abspath(ALLOWED_FILE_DIR))
        target = os.path.normpath(os.path.abspath(full_path))
        if not target.startswith(allowed):
            return f"Error: access denied. Only files under '{ALLOWED_FILE_DIR}' are allowed."

        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Successfully written to {path} ({len(content)} chars)"
        except Exception as e:
            return f"File write error: {e}"
