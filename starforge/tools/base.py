from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ..context import RuntimeContext
from ..observations import Observation


class Tool(Protocol):
    name: str
    description: str

    def run(self, arguments: dict[str, Any], context: RuntimeContext) -> Observation:
        ...


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    tool: Tool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = ToolSpec(
            name=tool.name,
            description=tool.description,
            tool=tool,
        )

    def extend(self, tools: list[Tool]) -> None:
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name].tool
        except KeyError as exc:
            raise KeyError(f"unknown tool: {name}") from exc

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def describe(self) -> list[dict[str, str]]:
        return [
            {"name": spec.name, "description": spec.description}
            for spec in sorted(self._tools.values(), key=lambda item: item.name)
        ]

    def execute(self, name: str, arguments: dict[str, Any], context: RuntimeContext) -> Observation:
        tool = self.get(name)
        return tool.run(arguments, context)
