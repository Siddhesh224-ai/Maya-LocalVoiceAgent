import unittest
from unittest.mock import patch

from livekit.agents import llm

from agent import _compact_chat_context


class ChatContextTest(unittest.TestCase):
    def test_context_keeps_instructions_and_recent_messages(self) -> None:
        context = llm.ChatContext()
        context.add_message(role="system", content="be concise")
        for index in range(12):
            context.add_message(role="user", content=f"message {index}")

        with patch.dict("os.environ", {"VOICE_HISTORY_MAX_ITEMS": "4"}):
            compact = _compact_chat_context(context)

        self.assertEqual(len(compact.items), 5)
        self.assertEqual(compact.items[0].role, "system")
        self.assertEqual(compact.items[-1].text_content, "message 11")

    def test_context_preserves_tool_exchange(self) -> None:
        context = llm.ChatContext()
        context.add_message(role="system", content="use tools")
        for index in range(9):
            context.add_message(role="user", content=f"message {index}")
        context.items.append(llm.FunctionCall(call_id="c1", name="search_knowledge_base", arguments="{}"))
        context.items.append(llm.FunctionCallOutput(call_id="c1", name="search_knowledge_base", output="facts", is_error=False))
        with patch.dict("os.environ", {"VOICE_HISTORY_MAX_ITEMS": "2"}):
            compact = _compact_chat_context(context)
        self.assertLess(len(compact.items), len(context.items))
        self.assertTrue(any(item.type == "function_call" for item in compact.items))
        self.assertTrue(any(item.type == "function_call_output" for item in compact.items))

