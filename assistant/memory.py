from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import ensure_dir, normalize_keywords, read_json, slugify, utc_now_iso, write_json, write_text


class MemoryStore:
    def __init__(self, blocks_dir: str | Path = "memory/blocks") -> None:
        self.blocks_dir = Path(blocks_dir)
        ensure_dir(self.blocks_dir)

    def _iter_blocks(self):
        for block_dir in sorted(self.blocks_dir.glob("*")):
            if not block_dir.is_dir():
                continue
            info_path = block_dir / "info.json"
            knowledge_path = block_dir / "knowledge.md"
            if info_path.exists() and knowledge_path.exists():
                yield block_dir, info_path, knowledge_path

    def find_in_memory(self, keywords: list[str], limit: int = 5) -> list[dict[str, Any]]:
        requested = set(normalize_keywords(keywords))
        if not requested:
            return []

        matches: list[dict[str, Any]] = []
        for block_dir, info_path, knowledge_path in self._iter_blocks():
            try:
                info = read_json(info_path)
                stored = set(normalize_keywords(info.get("keywords", [])))
            except Exception:
                continue

            overlap = requested & stored
            if not overlap:
                continue

            overlap_score = len(overlap)
            recall_score = len(overlap) / max(1, len(requested))
            score = overlap_score + recall_score

            knowledge = knowledge_path.read_text(encoding="utf-8")
            matches.append(
                {
                    "score": round(score, 3),
                    "overlap": sorted(overlap),
                    "name": info.get("name", block_dir.name),
                    "topic": info.get("topic", ""),
                    "keywords": info.get("keywords", []),
                    "dependencies": info.get("dependencies", []),
                    "source": info.get("source", []),
                    "version": info.get("version", "1.0.0"),
                    "created_at": info.get("created_at", ""),
                    "knowledge": knowledge,
                }
            )

        matches.sort(key=lambda x: (x["score"], len(x["overlap"])), reverse=True)
        return matches[:limit]

    def create_block(
        self,
        name: str,
        topic: str,
        keywords: list[str],
        knowledge: str,
        source: list[str],
        dependencies: list[str] | None = None,
    ) -> dict[str, Any]:
        block_name = slugify(name)
        block_dir = self.blocks_dir / block_name
        ensure_dir(block_dir)

        info = {
            "name": name,
            "topic": topic,
            "keywords": normalize_keywords(keywords),
            "dependencies": dependencies or [],
            "source": source,
            "version": "1.0.0",
            "created_at": utc_now_iso(),
        }

        info_path = block_dir / "info.json"
        knowledge_path = block_dir / "knowledge.md"

        write_json(info_path, info)
        write_text(knowledge_path, knowledge.strip() + "\n")

        return {
            "ok": True,
            "block": block_name,
            "info_path": str(info_path),
            "knowledge_path": str(knowledge_path),
        }
