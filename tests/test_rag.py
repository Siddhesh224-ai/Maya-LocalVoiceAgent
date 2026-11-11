import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rag import retrieve_with_timing


class RagRetrievalTest(unittest.TestCase):
    def test_retrieval_measures_embedding_and_database_work_separately(self) -> None:
        embeddings = SimpleNamespace(embed_query=lambda question: [0.1, 0.2])
        documents = [
            SimpleNamespace(
                page_content="A useful fact",
                metadata={"source": "guide.pdf", "page": 2},
            )
        ]
        store = SimpleNamespace(
            similarity_search_by_vector=lambda embedding, k: documents
        )
        with (
            patch("rag.get_embeddings", return_value=embeddings),
            patch("rag.get_vector_store", return_value=store),
        ):
            result = retrieve_with_timing("What is the useful fact?", limit=4)

        self.assertIn("A useful fact", result.context)
        self.assertEqual(result.result_count, 1)
        self.assertGreaterEqual(result.embedding_s, 0)
        self.assertGreaterEqual(result.database_s, 0)

    def test_retrieval_context_is_bounded(self) -> None:
        embeddings = SimpleNamespace(embed_query=lambda question: [0.1])
        documents = [SimpleNamespace(page_content="x" * 100, metadata={})]
        store = SimpleNamespace(similarity_search_by_vector=lambda embedding, k: documents)
        with (
            patch("rag.get_embeddings", return_value=embeddings),
            patch("rag.get_vector_store", return_value=store),
            patch.dict("os.environ", {"RAG_MAX_CONTEXT_CHARS": "25"}),
        ):
            result = retrieve_with_timing("question")

        self.assertEqual(len(result.context), 25)
