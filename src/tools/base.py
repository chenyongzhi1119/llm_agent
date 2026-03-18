"""Tool base class and registry"""

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """Base class for all tools"""

    name: str = ""
    description: str = ""
    parameters: dict = {}  # JSON Schema format parameter description

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool and return a string result"""
        ...

    def to_schema(self) -> dict:
        """Generate tool description for ReAct prompt"""
        params_desc = ""
        for param_name, param_info in self.parameters.items():
            required = "required" if param_info.get("required", False) else "optional"
            params_desc += f"    - {param_name} ({required}): {param_info.get('description', '')}\n"

        return {
            "name": self.name,
            "description": self.description,
            "parameters": params_desc.strip(),
        }


class ToolRegistry:
    """Tool registry that manages all available tools"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool"""
        if not tool.name:
            raise ValueError(f"Tool {tool.__class__.__name__} is missing 'name' attribute")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Get a tool by name"""
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        """Get all registered tools"""
        return list(self._tools.values())

    def get_tools_prompt(self) -> str:
        """Generate tool descriptions text for embedding in ReAct prompt"""
        lines = []
        for tool in self._tools.values():
            schema = tool.to_schema()
            lines.append(f"- **{schema['name']}**: {schema['description']}")
            if schema["parameters"]:
                lines.append(f"  Parameters:\n  {schema['parameters']}")
        return "\n".join(lines)
