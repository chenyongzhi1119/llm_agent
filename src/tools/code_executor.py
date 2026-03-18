"""Code executor tool - sandboxed Python execution with whitelist AST check"""

import ast
import subprocess
import sys

from src.tools.base import Tool
from config.settings import CODE_EXEC_TIMEOUT


# Modules explicitly allowed inside sandboxed code
ALLOWED_IMPORTS = {
    "math", "random", "statistics", "itertools", "functools",
    "collections", "string", "re", "json", "datetime", "time",
    "decimal", "fractions", "heapq", "bisect",
}


def _check_ast(code: str) -> str | None:
    """
    Walk the AST and reject any node that could escape the sandbox.
    Returns an error message string, or None if the code is safe.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"Syntax error: {e}"

    for node in ast.walk(tree):
        # Block all imports except the whitelist
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name.split(".")[0] for alias in node.names]
                if isinstance(node, ast.Import)
                else ([node.module.split(".")[0]] if node.module else [])
            )
            for name in names:
                if name not in ALLOWED_IMPORTS:
                    return f"Import '{name}' is not allowed. Allowed: {sorted(ALLOWED_IMPORTS)}"

        # Block attribute access on __builtins__ / __class__ / __subclasses__ etc.
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                return f"Dunder attribute access '{node.attr}' is not allowed."

        # Block calls to open(), eval(), exec(), compile(), __import__
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in {"open", "eval", "exec", "compile", "__import__"}:
                    return f"Call to '{node.func.id}()' is not allowed."
            # Block subscript access on __builtins__ e.g. __builtins__['open']
            if isinstance(node.func, ast.Subscript):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "__builtins__":
                    return "Access to '__builtins__' subscript is not allowed."

    return None


class CodeExecutorTool(Tool):
    name = "code_executor"
    description = "Run Python code in a sandbox. Returns stdout/stderr. Only math/data/stdlib allowed."
    parameters = {
        "code": {"description": "Python code to run", "required": True},
    }

    def execute(self, **kwargs) -> str:
        code = kwargs.get("code", "")
        if not code:
            return "Error: please provide code to execute"

        # AST whitelist check (stronger than string matching)
        err = _check_ast(code)
        if err:
            return f"Security check failed: {err}"

        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=CODE_EXEC_TIMEOUT,
            )

            output = ""
            if result.stdout:
                output += f"stdout:\n{result.stdout}"
            if result.stderr:
                output += f"stderr:\n{result.stderr}"
            return output.strip() or "(no output)"

        except subprocess.TimeoutExpired:
            return f"Error: execution timed out after {CODE_EXEC_TIMEOUT}s"
        except Exception as e:
            return f"Execution error: {e}"
