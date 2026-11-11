import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from services.voice_agent_service import AGENT_READY_ATTRIBUTE, wait_for_agent_ready


class _RoomAPI:
    def __init__(self, responses):
        self._responses = iter(responses)
        self.list_participants = AsyncMock(side_effect=self._next_response)

    async def _next_response(self, _request):
        return next(self._responses)


class AgentReadinessTest(unittest.IsolatedAsyncioTestCase):
    async def test_wait_for_agent_ready_polls_until_attribute_is_published(self) -> None:
        room = _RoomAPI(
            [
                SimpleNamespace(participants=[]),
                SimpleNamespace(
                    participants=[
                        SimpleNamespace(attributes={AGENT_READY_ATTRIBUTE: "true"})
                    ]
                ),
            ]
        )
        lkapi = SimpleNamespace(room=room)

        ready = await wait_for_agent_ready(
            lkapi,
            "call-test",
            timeout_s=1.0,
            poll_interval_s=0.0,
        )

        self.assertEqual(ready.attributes[AGENT_READY_ATTRIBUTE], "true")
        self.assertEqual(room.list_participants.await_count, 2)

    async def test_token_is_created_only_after_dispatch_is_ready(self) -> None:
        events: list[str] = []
        lkapi = SimpleNamespace(
            room=SimpleNamespace(),
            agent_dispatch=SimpleNamespace(
                create_dispatch=AsyncMock(side_effect=lambda _request: events.append("dispatch"))
            ),
            aclose=AsyncMock(),
        )

        async def mark_ready(_lkapi, _room_name):
            events.append("ready")

        with (
            patch("services.voice_agent_service.get_lkapi", return_value=lkapi),
            patch("services.voice_agent_service.create_room", new=AsyncMock()),
            patch("services.voice_agent_service.wait_for_agent_ready", new=mark_ready),
            patch(
                "services.voice_agent_service.generate_token",
                side_effect=lambda **_kwargs: events.append("token") or "jwt",
            ),
            patch.dict("os.environ", {"LIVEKIT_URL": "wss://example.test"}, clear=False),
        ):
            from services.voice_agent_service import VoiceAgentService

            response = await VoiceAgentService().start_call()

        self.assertEqual(events, ["dispatch", "ready", "token"])
        self.assertEqual(response.token, "jwt")
        lkapi.aclose.assert_awaited_once()
