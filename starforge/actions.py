from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .observations import Observation


@dataclass(slots=True)
class ActionRequest:
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


@dataclass(slots=True)
class ActionRecord:
    tool: str
    arguments: dict[str, Any]
    observation: Observation
    status: str = "completed"
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "arguments": dict(self.arguments),
            "status": self.status,
            "rationale": self.rationale,
            "observation": self.observation.to_dict(),
        }
