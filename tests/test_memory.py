import shutil
import unittest
from pathlib import Path

from assistant.memory import MemoryStore


class TestMemoryStore(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("test_memory_blocks")
        self.store = MemoryStore(blocks_dir=self.test_dir)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_create_and_find_block(self):
        self.store.create_block(
            name="Test Block",
            topic="Testing",
            keywords=["unittest", "python"],
            knowledge="This is a test knowledge block.",
            source=["http://example.com"],
        )

        # Search by keyword
        results = self.store.find_in_memory(["unittest"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "Test Block")
        self.assertEqual(results[0]["knowledge"], "This is a test knowledge block.")

    def test_find_no_match(self):
        self.store.create_block(
            name="Test Block",
            topic="Testing",
            keywords=["unittest"],
            knowledge="...",
            source=[],
        )
        results = self.store.find_in_memory(["nonexistent"])
        self.assertEqual(len(results), 0)

    def test_multiple_keywords_scoring(self):
        self.store.create_block(
            name="Block A",
            topic="T1",
            keywords=["apple", "banana"],
            knowledge="...",
            source=[],
        )
        self.store.create_block(
            name="Block B",
            topic="T2",
            keywords=["apple", "cherry"],
            knowledge="...",
            source=[],
        )

        results = self.store.find_in_memory(["apple", "banana"])
        self.assertEqual(len(results), 2)
        self.assertEqual(
            results[0]["name"], "Block A"
        )  # Higher score due to more keyword matches

    def test_semantic_search(self):
        self.store.create_block(
            name="Reliability Notes",
            topic="Tooling",
            keywords=["network", "stability"],
            knowledge="Use retry and fallback strategy when tool calls fail with timeout or rate limit.",
            source=[],
        )
        results = self.store.semantic_search("retry fallback for timeout failures", limit=3)
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "Reliability Notes")
        self.assertEqual(results[0]["match_type"], "semantic")

    def test_find_in_memory_semantic_fallback(self):
        self.store.create_block(
            name="Execution Strategy",
            topic="Agent",
            keywords=["planner"],
            knowledge="Task reflection improves success by reviewing each tool result and retrying when needed.",
            source=[],
        )
        # Query keyword is not in block keywords, but is semantically present in knowledge.
        results = self.store.find_in_memory(["reflection"])
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "Execution Strategy")

    def test_memory_feedback_stats(self):
        self.store.create_block(
            name="Bug Fix Pattern",
            topic="Python",
            keywords=["import", "error"],
            knowledge="Fix import path by checking package __init__ and PYTHONPATH.",
            source=[],
        )
        fb = self.store.record_feedback(
            block_name="bug_fix_pattern",
            success=True,
            confidence=0.9,
            source="test",
        )
        self.assertTrue(fb["ok"])
        results = self.store.find_in_memory(["import"])
        self.assertGreaterEqual(len(results), 1)
        self.assertIn("block", results[0])


if __name__ == "__main__":
    unittest.main()
