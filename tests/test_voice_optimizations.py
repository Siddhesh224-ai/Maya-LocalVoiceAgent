import asyncio
import unittest
from unittest.mock import patch

from livekit.agents import llm

from agent import RagAgent, TurnLatencyTable, _compact_chat_context, _pop_chunk
from moonshine_streaming_stt import _TranscriptAccumulator
from rag import RetrievalResult


class FirstChunkPolicyTest(unittest.TestCase):
    def test_first_chunk_flushes_at_clause_after_four_words(self) -> None:
        chunk, remainder = _pop_chunk("I can help today, and I will check that.", first=True)

        self.assertEqual(chunk, "I can help today,")
        self.assertEqual(remainder, "and I will check that.")

    def test_first_chunk_falls_back_to_eight_words(self) -> None:
        chunk, remainder = _pop_chunk(
            "I can help with this request today please continue now.", first=True
        )

        self.assertEqual(chunk, "I can help with this request today please")
        self.assertEqual(remainder, "continue now.")

    def test_final_flush_returns_all_remaining_text(self) -> None:
        chunk, remainder = _pop_chunk("A final unpunctuated answer", final=True)

        self.assertEqual(chunk, "A final unpunctuated answer")
        self.assertEqual(remainder, "")

    def test_first_chunk_caps_long_sentence(self) -> None:
        chunk, remainder = _pop_chunk(
            "The incomprehensibly overcomplicated explanation contains disproportionately convoluted terminology immediately today.",
            first=True,
        )

        self.assertLessEqual(len(chunk.split()), 8)
        self.assertGreaterEqual(len(chunk.split()), 4)
        self.assertTrue(remainder)


class TranscriptAccumulatorTest(unittest.TestCase):
    def test_interim_and_completed_lines_keep_the_full_utterance(self) -> None:
        accumulator = _TranscriptAccumulator()

        self.assertEqual(accumulator.interim("first line"), "first line")
        self.assertEqual(accumulator.complete("first line"), ("first line", "first line"))
        self.assertEqual(accumulator.interim("second line"), "first line second line")
        self.assertEqual(
            accumulator.complete("second line"),
            ("second line", "first line second line"),
        )
        self.assertEqual(accumulator.full, "first line second line")


class ChatContextCompactionTest(unittest.TestCase):
    def test_chunked_trim_strips_old_rag_notes(self) -> None:
        context = llm.ChatContext()
        context.add_message(role="system", content="stable instructions")
        context.add_message(role="system", content="Relevant notes: old retrieval")
        for index in range(13):
            context.add_message(role="user", content=f"message {index}")

        with patch.dict("os.environ", {"VOICE_HISTORY_MAX_ITEMS": "4"}):
            compact = _compact_chat_context(context)

        self.assertFalse(any("Relevant notes:" in item.text_content for item in compact.messages()))
        self.assertEqual(compact.messages()[-1].text_content, "message 12")
        self.assertEqual(len([item for item in compact.messages() if item.role == "user"]), 4)

    def test_current_turn_rag_note_is_kept(self) -> None:
        context = llm.ChatContext()
        context.add_message(role="system", content="stable instructions")
        context.add_message(role="user", content="What is covered?")
        context.add_message(role="system", content="Relevant notes: current retrieval")

        compact = _compact_chat_context(context)

        self.assertTrue(any("current retrieval" in item.text_content for item in compact.messages()))


class RagGateTest(unittest.TestCase):
    def test_relevance_gate_uses_pgvector_distance_scores(self) -> None:
        relevant = RetrievalResult("notes", 0.01, 0.01, 1, scores=(0.25,))
        irrelevant = RetrievalResult("notes", 0.01, 0.01, 1, scores=(0.9,))

        with patch.dict("os.environ", {"RAG_MAX_DISTANCE": "0.65"}):
            self.assertTrue(relevant.is_relevant)
            self.assertFalse(irrelevant.is_relevant)

    def test_retrieval_timeout_skips_context_injection(self) -> None:
        async def exercise() -> llm.ChatContext:
            agent = RagAgent("instructions", TurnLatencyTable())
            turn_ctx = llm.ChatContext()
            message = llm.ChatMessage(role="user", content=["indexed roadmap question"])

            async def slow_retrieval(_question: str):
                await asyncio.sleep(1.0)

            with (
                patch("agent._rag_enabled", return_value=True),
                patch.object(agent, "_retrieve_for_turn", side_effect=slow_retrieval),
            ):
                await agent.on_user_turn_completed(turn_ctx, message)
            return turn_ctx

        context = asyncio.run(exercise())
        self.assertFalse(any("Relevant notes:" in item.text_content for item in context.messages()))

