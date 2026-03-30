from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .tools import (
    HttpRequestTool,
    ListFilesTool,
    ReadWebpageTool,
    ReadFileTool,
    RunCommandTool,
    SearchTool,
    ToolRegistry,
    WebSearchTool,
    WriteFileTool,
)


@dataclass(slots=True)
class Adapter:
    name: str
    description: str
    tools: list[object] = field(default_factory=list)
    defaults: dict[str, Any] = field(default_factory=dict)

    def configure(self, registry: ToolRegistry) -> None:
        for tool in self.tools:
            registry.register(tool)

    def merge_context(self, payload: dict[str, Any]) -> dict[str, Any]:
        merged = dict(self.defaults)
        merged.update(payload)
        return merged


class CLIAdapter(Adapter):
    def __init__(self) -> None:
        super().__init__(
            name="cli",
            description="General-purpose command-line runtime.",
            tools=[
                RunCommandTool(),
                ReadFileTool(),
                WriteFileTool(),
                ListFilesTool(),
                WebSearchTool(),
                ReadWebpageTool(),
                SearchTool(),
            ],
        )


class CodeProjectAdapter(Adapter):
    def __init__(self) -> None:
        super().__init__(
            name="code",
            description="Code-project environment with filesystem and CLI tools.",
            tools=[
                RunCommandTool(),
                ReadFileTool(),
                WriteFileTool(),
                ListFilesTool(),
                WebSearchTool(),
                ReadWebpageTool(),
                SearchTool(),
            ],
        )

    def merge_context(self, payload: dict[str, Any]) -> dict[str, Any]:
        merged = super().merge_context(payload)
        commands = list(merged.get("commands", []) or [])
        for key in ("diagnostic_command", "test_command", "build_command"):
            value = merged.get(key)
            if value and value not in commands:
                commands.append(value)
        if commands:
            merged["commands"] = commands
        return merged


class APIWorkflowAdapter(Adapter):
    def __init__(self) -> None:
        super().__init__(
            name="api",
            description="API and research workflow environment.",
            tools=[
                HttpRequestTool(),
                WebSearchTool(),
                ReadWebpageTool(),
                SearchTool(),
                WriteFileTool(),
                ReadFileTool(),
            ],
        )


def get_adapter(name: str | None) -> Adapter:
    key = str(name or "cli").strip().lower()
    if key == "code":
        return CodeProjectAdapter()
    if key == "api":
        return APIWorkflowAdapter()
    return CLIAdapter()
