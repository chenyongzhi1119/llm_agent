"""Unit tests for core components"""

import pytest
from src.agent.parser import parse_react_output, ReActAction, ReActFinish, _strip_fabricated_observation
from src.tools.base import Tool, ToolRegistry
from src.tools.calculator import CalculatorTool
from src.tools.code_executor import CodeExecutorTool, _check_ast
from src.memory.short_term import ShortTermMemory


# ── parser ────────────────────────────────────────────────────────────────────

class TestParser:
    def test_parse_final_answer(self):
        text = "Thought: done\nFinal Answer: Paris"
        result = parse_react_output(text)
        assert isinstance(result, ReActFinish)
        assert result.answer == "Paris"

    def test_parse_action(self):
        text = 'Thought: need calc\nAction: calculator\nAction Input: {"expression": "1+1"}'
        result = parse_react_output(text)
        assert isinstance(result, ReActAction)
        assert result.tool_name == "calculator"
        assert result.tool_input == {"expression": "1+1"}

    def test_strip_fabricated_observation(self):
        text = "Thought: I'll calculate\nAction: calculator\nAction Input: {}\nObservation: 42\nFinal Answer: 42"
        stripped = _strip_fabricated_observation(text)
        assert "Observation" not in stripped
        assert "Action: calculator" in stripped

    def test_fabricated_observation_forces_tool_call(self):
        # LLM fabricates observation — should still extract the Action, not the fake observation
        text = 'Thought: calc\nAction: calculator\nAction Input: {"expression": "2+2"}\nObservation: 4'
        result = parse_react_output(text)
        assert isinstance(result, ReActAction)
        assert result.tool_name == "calculator"

    def test_invalid_action_input_falls_back(self):
        text = "Thought: ok\nAction: calculator\nAction Input: not json at all"
        result = parse_react_output(text)
        assert isinstance(result, ReActAction)
        assert "input" in result.tool_input

    def test_no_action_no_final_answer_returns_finish(self):
        text = "Thought: I already know the answer is Paris"
        result = parse_react_output(text)
        assert isinstance(result, ReActFinish)


# ── ToolRegistry ──────────────────────────────────────────────────────────────

class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = CalculatorTool()
        registry.register(tool)
        assert registry.get("calculator") is tool

    def test_get_missing_returns_none(self):
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    def test_list_tools(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        tools = registry.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "calculator"

    def test_register_nameless_tool_raises(self):
        class BadTool(Tool):
            name = ""
            description = "bad"
            parameters = {}
            def execute(self, **kwargs): return ""

        with pytest.raises(ValueError):
            ToolRegistry().register(BadTool())

    def test_get_tools_prompt_contains_names(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        prompt = registry.get_tools_prompt()
        assert "calculator" in prompt


# ── CalculatorTool ────────────────────────────────────────────────────────────

class TestCalculatorTool:
    def setup_method(self):
        self.tool = CalculatorTool()

    def test_basic_math(self):
        assert self.tool.execute(expression="2 + 2") == "4"

    def test_complex_expression(self):
        assert self.tool.execute(expression="(128 * 37) + 512") == "5248"

    def test_division(self):
        assert self.tool.execute(expression="10 / 4") == "2.5"

    def test_empty_expression(self):
        result = self.tool.execute(expression="")
        assert "Error" in result

    def test_blocked_chars(self):
        result = self.tool.execute(expression="__import__('os')")
        assert "Error" in result


# ── CodeExecutorTool (AST whitelist) ─────────────────────────────────────────

class TestCodeExecutorAST:
    def test_safe_code_passes(self):
        assert _check_ast("print(1 + 2)") is None

    def test_import_math_allowed(self):
        assert _check_ast("import math\nprint(math.pi)") is None

    def test_import_os_blocked(self):
        err = _check_ast("import os")
        assert err is not None
        assert "os" in err

    def test_open_call_blocked(self):
        err = _check_ast("open('secret.txt')")
        assert err is not None

    def test_dunder_attribute_blocked(self):
        err = _check_ast("x.__class__.__subclasses__()")
        assert err is not None

    def test_builtins_bypass_blocked(self):
        # Previously bypassed blacklist: __builtins__['open']
        err = _check_ast("__builtins__['open']('file.txt')")
        assert err is not None

    def test_exec_blocked(self):
        err = _check_ast("exec('import os')")
        assert err is not None


class TestCodeExecutorRun:
    def setup_method(self):
        self.tool = CodeExecutorTool()

    def test_print_output(self):
        result = self.tool.execute(code="print(42)")
        assert "42" in result

    def test_list_comprehension(self):
        result = self.tool.execute(code="print([x**2 for x in range(5)])")
        assert "16" in result

    def test_import_os_blocked(self):
        result = self.tool.execute(code="import os")
        assert "Security check failed" in result or "Error" in result


# ── ShortTermMemory ───────────────────────────────────────────────────────────

class TestShortTermMemory:
    def test_add_and_get(self):
        mem = ShortTermMemory(max_messages=10)
        mem.add("user", "hello")
        mem.add("assistant", "hi")
        msgs = mem.get_messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"

    def test_system_prompt_stays_at_index_0(self):
        mem = ShortTermMemory(max_messages=10)
        mem.set_system_prompt("You are a bot")
        mem.add("user", "hello")
        msgs = mem.get_messages()
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are a bot"

    def test_sliding_window_trims_oldest(self):
        mem = ShortTermMemory(max_messages=4)
        mem.set_system_prompt("sys")
        for i in range(5):
            mem.add("user", f"msg{i}")
        msgs = mem.get_messages()
        # system + 3 most recent user messages = 4 total
        assert len(msgs) == 4
        assert msgs[0]["role"] == "system"
        assert "msg4" in msgs[-1]["content"]

    def test_clear_keeps_system_prompt(self):
        mem = ShortTermMemory()
        mem.set_system_prompt("sys")
        mem.add("user", "hello")
        mem.clear()
        msgs = mem.get_messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"

    def test_len(self):
        mem = ShortTermMemory()
        mem.add("user", "a")
        mem.add("assistant", "b")
        assert len(mem) == 2
