from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from hashlib import blake2b
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class MemoryPattern:
    pattern_type: str
    context: str
    resolution_strategy: str
    confidence: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_type": self.pattern_type,
            "context": self.context,
            "resolution_strategy": self.resolution_strategy,
            "confidence": round(max(0.0, min(self.confidence, 1.0)), 3),
            "metadata": dict(self.metadata),
        }


class MemoryStore:
    def __init__(self, root: str | Path | None = None, embedding_dims: int = 256) -> None:
        self.root = Path(root or Path.home() / ".starforge").expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.patterns_path = self.root / "patterns.jsonl"
        self.embedding_dims = max(64, min(int(embedding_dims), 2048))
        self._patterns: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        self._patterns = []
        if not self.patterns_path.exists():
            return
        for line in self.patterns_path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                self._patterns.append(item)

    def remember(
        self,
        pattern_type: str,
        context: str,
        resolution_strategy: str,
        confidence: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        pattern = MemoryPattern(
            pattern_type=pattern_type,
            context=context,
            resolution_strategy=resolution_strategy,
            confidence=confidence,
            metadata=metadata or {},
        )
        payload = pattern.to_dict()
        with self.patterns_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        self._patterns.append(payload)
        return payload

    def search(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        query_embedding = self._embed(query)
        ranked: list[tuple[float, dict[str, Any]]] = []
        for item in self._patterns:
            haystack = " ".join(
                [
                    str(item.get("pattern_type", "")),
                    str(item.get("context", "")),
                    str(item.get("resolution_strategy", "")),
                    json.dumps(item.get("metadata", {}), sort_keys=True),
                ]
            )
            score = self._cosine(query_embedding, self._embed(haystack))
            if score <= 0:
                continue
            ranked.append((score, item))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        results: list[dict[str, Any]] = []
        for score, item in ranked[: max(1, limit)]:
            enriched = dict(item)
            enriched["score"] = round(score, 3)
            results.append(enriched)
        return results

    def _embed(self, text: str) -> dict[int, float]:
        counts: dict[int, float] = {}
        for token in re.findall(r"[a-zA-Z0-9_]{2,}", text.lower()):
            digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest, "little") % self.embedding_dims
            counts[bucket] = counts.get(bucket, 0.0) + 1.0
        norm = math.sqrt(sum(value * value for value in counts.values()))
        if norm == 0:
            return {}
        return {bucket: value / norm for bucket, value in counts.items()}

    @staticmethod
    def _cosine(left: dict[int, float], right: dict[int, float]) -> float:
        if not left or not right:
            return 0.0
        if len(left) > len(right):
            left, right = right, left
        return sum(weight * right.get(bucket, 0.0) for bucket, weight in left.items())
