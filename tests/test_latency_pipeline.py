import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import rag
from agent import RAG_INSTRUCTIONS, VoiceTrace, _pop_completed_sentence


class RagReadinessTest(unittest.TestCase):
    def setUp(self) -> None:
        rag.reset_rag_resources_for_testing()

    def tearDown(self) -> None:
        rag.reset_rag_resources_for_testing()

    def test_warmup_and_retrieval_share_one_embedding_client_and_store(self) -> None:
        created_embeddings = 0
        created_stores = 0
        creation_lock = threading.Lock()

        class Embeddings:
            def embed_query(self, _question):
                return [0.1, 0.2]

        class Store:
            def similarity_search_by_vector(self, _embedding, k):
                return []

        def make_embeddings():
            nonlocal created_embeddings
            time.sleep(0.02)
            with creation_lock:
                created_embeddings += 1
            return Embeddings()

        def make_store(_embeddings):
            nonlocal created_stores
            time.sleep(0.02)
            with creation_lock:
                created_stores += 1
            return Store()

        with (
            patch("rag._create_embeddings", side_effect=make_embeddings),
            patch("rag._create_vector_store", side_effect=make_store),
        ):
            with ThreadPoolExecutor(max_workers=2) as executor:
                warmup = executor.submit(rag.ensure_rag_ready)
                retrieval = executor.submit(rag.retrieve_with_timing, "What is the roadmap?")
                warmup.result()
                retrieval.result()

        self.assertEqual(created_embeddings, 1)
        self.assertEqual(created_stores, 1)


class GroundingContractTest(unittest.TestCase):
    def test_direct_answer_profile_keeps_tool_disabled(self) -> None:
        self.assertIn("Answer every question directly", RAG_INSTRUCTIONS)
        self.assertNotIn("query_knowledge_base", RAG_INSTRUCTIONS)
        self.assertIn("Do not expand or define an acronym", RAG_INSTRUCTIONS)
        self.assertIn("at most two short sentences", RAG_INSTRUCTIONS)


class VoiceTraceTest(unittest.TestCase):
    def test_records_ordered_steps_and_durations(self) -> None:
        trace = VoiceTrace("turn-1", clock=iter([10.0, 10.25, 10.75]).__next__)

        trace.mark("speech_started")
        trace.mark("stt_final")
        trace.mark("first_audio_published")

        self.assertEqual(trace.turn_id, "turn-1")
        self.assertEqual(trace.steps, ["speech_started", "stt_final", "first_audio_published"])
        self.assertEqual(trace.elapsed("speech_started", "stt_final"), 0.25)
        self.assertEqual(trace.elapsed("stt_final", "first_audio_published"), 0.5)
        self.assertEqual(
            trace.snapshot()["durations"],
            {
                "speech_started_to_stt_final": 0.25,
                "stt_final_to_first_audio_published": 0.5,
            },
        )

    def test_mark_is_idempotent_for_first_timestamp(self) -> None:
        trace = VoiceTrace("turn-1", clock=iter([1.0, 2.0, 3.0]).__next__)

        trace.mark("llm_first_token")
        trace.mark("llm_first_token")

        self.assertEqual(trace.steps, ["llm_first_token"])
        self.assertEqual(trace.marks["llm_first_token"], 1.0)


class SentenceFlushTest(unittest.TestCase):
    def test_keeps_short_fragment_buffered_until_a_real_sentence(self) -> None:
        sentence, remainder = _pop_completed_sentence("Hi. The latency is low. Next")

        self.assertEqual(sentence, "Hi. The latency is low.")
        self.assertEqual(remainder, "Next")

    def test_final_flush_returns_unpunctuated_tail(self) -> None:
        sentence, remainder = _pop_completed_sentence("A complete answer", final=True)

        self.assertEqual(sentence, "A complete answer")
        self.assertEqual(remainder, "")

    def test_flushes_sentence_that_ends_at_current_chunk_boundary(self) -> None:
        sentence, remainder = _pop_completed_sentence(
            "This sentence is long enough to flush."
        )

        self.assertEqual(sentence, "This sentence is long enough to flush.")
        self.assertEqual(remainder, "")



