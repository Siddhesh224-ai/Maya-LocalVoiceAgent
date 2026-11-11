import unittest

from local_llm_client import PunctuationChunker


class PunctuationChunkerTest(unittest.TestCase):
    def test_emits_punctuated_phrases_immediately(self) -> None:
        chunker = PunctuationChunker()
        self.assertEqual(chunker.push("Hello, world"), ["Hello,"])
        self.assertEqual(chunker.push("! Next"), ["world!"])
        self.assertEqual(chunker.flush(), "Next")

