"""Calculator tool - example tool for verifying ReAct flow"""

from src.tools.base import Tool


class CalculatorTool(Tool):
    name = "calculator"
    description = "Math calculator. Input a math expression string and get the result."
    parameters = {
        "expression": {
            "description": "Math expression, e.g. '2 + 3 * 4'",
            "required": True,
        }
    }

    def execute(self, **kwargs) -> str:
        expression = kwargs.get("expression", "")
        if not expression:
            return "Error: please provide a math expression"

        # Safety check: only allow digits and basic operators
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "Error: expression contains disallowed characters, only digits and +-*/(). are supported"

        try:
            result = eval(expression)  # Safe: character whitelist already applied
            return str(result)
        except Exception as e:
            return f"Calculation error: {e}"
