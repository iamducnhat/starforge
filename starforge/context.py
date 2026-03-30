from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RuntimeContext:
    objective: str
    working_dir: Path
    constraints: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(
        cls,
        objective: str,
        context: dict[str, Any] | None = None,
    ) -> "RuntimeContext":
        payload = dict(context or {})
        raw_working_dir = payload.pop("working_dir", ".")
        working_dir = Path(raw_working_dir).expanduser().resolve()
        constraints = list(payload.pop("constraints", []) or [])
        return cls(
            objective=objective,
            working_dir=working_dir,
            constraints=constraints,
            metadata=payload,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "working_dir": str(self.working_dir),
            "constraints": list(self.constraints),
            **self.metadata,
        }
