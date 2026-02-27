from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from .utils import ensure_dir, normalize_keywords, read_json, slugify, utc_now_iso, write_json, write_text


class MemoryStore:
    def __init__(self, blocks_dir: str | Path = "memory/blocks") -> None:
        self.blocks_dir = Path(blocks_dir)
        ensure_dir(self.blocks_dir)
        self._metadata_cache: dict[str, dict[str, Any]] = {}
        self._load_metadata() # Initial load

    def _load_metadata(self) -> None:
        """Load or refresh the metadata cache from disk."""
        new_cache = {}
        for block_dir, info_path, knowledge_path in self._iter_blocks():
            try:
                info = read_json(info_path)
                # Store paths and normalized keywords for fast lookup
                info["_block_dir"] = block_dir
                info["_knowledge_path"] = knowledge_path
                info["_norm_keywords"] = set(normalize_keywords(info.get("keywords", [])))
                new_cache[block_dir.name] = info
            except Exception:
                continue
        self._metadata_cache = new_cache

    def _iter_blocks(self) -> Iterator[tuple[Path, Path, Path]]:
        """Iterate over valid memory blocks, yielding (block_dir, info_path, knowledge_path)."""
        for block_dir in sorted(self.blocks_dir.glob("*")):
            if not block_dir.is_dir():
                continue
            info_path = block_dir / "info.json"
            knowledge_path = block_dir / "knowledge.md"
            if info_path.exists() and knowledge_path.exists():
                yield block_dir, info_path, knowledge_path

    def find_in_memory(self, keywords: list[str], limit: int = 5) -> list[dict[str, Any]]:
        """
        Search for memory blocks matching the given keywords.
        
        Returns a list of matches sorted by relevance score.
        """
        requested = set(normalize_keywords(keywords))
        if not requested:
            return []

        scored_matches: list[dict[str, Any]] = []
        for block_name, info in self._metadata_cache.items():
            stored = info.get("_norm_keywords", set())
            overlap = requested & stored
            if not overlap:
                continue

            overlap_score = len(overlap)
            recall_score = len(overlap) / max(1, len(requested))
            score = overlap_score + recall_score

            scored_matches.append({
                "score": round(score, 3),
                "overlap": sorted(overlap),
                "info": info,
            })

        # Sort by score and overlap size
        scored_matches.sort(key=lambda x: (x["score"], len(x["overlap"])), reverse=True)
        
        # Only load knowledge for the top matches
        results: list[dict[str, Any]] = []
        for match in scored_matches[:limit]:
            info = match["info"]
            knowledge_path = info["_knowledge_path"]
            try:
                knowledge = knowledge_path.read_text(encoding="utf-8")
            except Exception:
                knowledge = "[Error reading knowledge content]"

            results.append({
                "score": match["score"],
                "overlap": match["overlap"],
                "name": info.get("name", info["_block_dir"].name),
                "topic": info.get("topic", ""),
                "keywords": info.get("keywords", []),
                "dependencies": info.get("dependencies", []),
                "source": info.get("source", []),
                "version": info.get("version", "1.0.0"),
                "created_at": info.get("created_at", ""),
                "knowledge": knowledge,
            })

        return results

    def create_block(
        self,
        name: str,
        topic: str,
        keywords: list[str],
        knowledge: str,
        source: list[str],
        dependencies: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Create a new memory block with the given metadata and knowledge content.
        
        Args:
            name: Human-readable name of the block.
            topic: General topic area.
            keywords: List of keywords for retrieval.
            knowledge: Markdown content of the knowledge block.
            source: List of source URLs or references.
            dependencies: Optional list of other block names this depends on.
        """
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

        # Update cache
        info_copy = info.copy()
        info_copy["_block_dir"] = block_dir
        info_copy["_knowledge_path"] = knowledge_path
        info_copy["_norm_keywords"] = set(info["keywords"])
        self._metadata_cache[block_name] = info_copy

        return {
            "ok": True,
            "block": block_name,
            "info_path": str(info_path),
            "knowledge_path": str(knowledge_path),
        }
