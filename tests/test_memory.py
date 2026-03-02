import unittest
import shutil
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
            source=["http://example.com"]
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
            source=[]
        )
        results = self.store.find_in_memory(["nonexistent"])
        self.assertEqual(len(results), 0)

    def test_multiple_keywords_scoring(self):
        self.store.create_block(
            name="Block A",
            topic="T1",
            keywords=["apple", "banana"],
            knowledge="...",
            source=[]
        )
        self.store.create_block(
            name="Block B",
            topic="T2",
            keywords=["apple", "cherry"],
            knowledge="...",
            source=[]
        )
        
        results = self.store.find_in_memory(["apple", "banana"])
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["name"], "Block A") # Higher score due to more keyword matches

if __name__ == "__main__":
    unittest.main()