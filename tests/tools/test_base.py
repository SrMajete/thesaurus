"""Tests for the Tool Protocol, ToolName enum, and API formatting."""

from dataclasses import dataclass
from typing import Any

from python_agent.tools.base import (
    REASON_FIELD_DESCRIPTION,
    ToolName,
    find_tool,
    tools_to_api_format,
)


@dataclass
class FakeTool:
    """Minimal Tool-compatible object for testing."""

    name: str = "fake_tool"
    description: str = "A fake tool for tests"
    needs_permission: bool = False
    is_parallelizable: bool = True
    is_intercepted: bool = False
    input_schema: dict[str, Any] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.input_schema is None:
            self.input_schema = {
                "type": "object",
                "properties": {"x": {"type": "string", "description": "x"}},
                "required": ["x"],
            }

    async def execute(self, **params: Any) -> str:
        return "ok"


class TestToolName:
    def test_is_strenum(self) -> None:
        assert isinstance(ToolName.READ_FILE, str)
        assert ToolName.READ_FILE == "read_file"

    def test_all_values_are_snake_case(self) -> None:
        import re
        pattern = re.compile(r"^[a-z]+(?:_[a-z]+)+$")
        for member in ToolName:
            assert pattern.match(member.value), f"Bad name: {member.value}"

    def test_uniqueness(self) -> None:
        values = [m.value for m in ToolName]
        assert len(values) == len(set(values))


class TestFindTool:
    def test_found(self) -> None:
        a = FakeTool(name="a")
        b = FakeTool(name="b")
        assert find_tool("b", [a, b]) is b

    def test_not_found_returns_none(self) -> None:
        assert find_tool("missing", [FakeTool(name="a")]) is None

    def test_empty_list_returns_none(self) -> None:
        assert find_tool("a", []) is None


class TestToolsToApiFormat:
    def test_empty_list(self) -> None:
        assert tools_to_api_format([]) == []

    def test_includes_name_and_description(self) -> None:
        t = FakeTool()
        formatted = tools_to_api_format([t])
        assert formatted[0]["name"] == t.name
        assert formatted[0]["description"] == t.description

    def test_reason_injected_as_first_property(self) -> None:
        t = FakeTool()
        formatted = tools_to_api_format([t])
        props = formatted[0]["input_schema"]["properties"]
        first_key = next(iter(props))
        assert first_key == "reason"
        assert props["reason"]["description"] == REASON_FIELD_DESCRIPTION

    def test_reason_added_to_required(self) -> None:
        t = FakeTool()
        formatted = tools_to_api_format([t])
        required = formatted[0]["input_schema"]["required"]
        assert required[0] == "reason"
        assert "x" in required  # existing required preserved

    def test_last_tool_has_cache_control(self) -> None:
        a = FakeTool(name="a")
        b = FakeTool(name="b")
        formatted = tools_to_api_format([a, b])
        assert "cache_control" not in formatted[0]
        assert formatted[1]["cache_control"] == {"type": "ephemeral"}

    def test_schema_is_deep_copied(self) -> None:
        """Mutating the formatted schema must not affect the original tool."""
        t = FakeTool()
        original_props = dict(t.input_schema["properties"])
        formatted = tools_to_api_format([t])
        formatted[0]["input_schema"]["properties"]["x"]["description"] = "changed"
        assert t.input_schema["properties"] == original_props

    def test_reason_not_duplicated_when_already_required(self) -> None:
        t = FakeTool()
        t.input_schema["required"] = ["reason", "x"]
        formatted = tools_to_api_format([t])
        assert formatted[0]["input_schema"]["required"].count("reason") == 1

    def test_schema_without_required(self) -> None:
        t = FakeTool()
        t.input_schema = {"type": "object", "properties": {}}
        formatted = tools_to_api_format([t])
        assert formatted[0]["input_schema"]["required"] == ["reason"]

    def test_schema_without_properties(self) -> None:
        t = FakeTool()
        t.input_schema = {"type": "object"}
        formatted = tools_to_api_format([t])
        assert "reason" in formatted[0]["input_schema"]["properties"]
