import asyncio
import os
import uuid

from livekit import api

from schemas.call_start_response import StartCallResponse

from token_generation_livekti import generate_token, get_lkapi, create_room


AGENT_READY_ATTRIBUTE = "voice_agent.ready"
DEFAULT_AGENT_READY_TIMEOUT_S = 45.0
DEFAULT_AGENT_READY_POLL_S = 0.1


async def wait_for_agent_ready(
    lkapi,
    room_name: str,
    *,
    timeout_s: float | None = None,
    poll_interval_s: float | None = None,
):
    """Wait until the dispatched agent reports that its session is ready.

    Dispatch is asynchronous: creating the dispatch only asks LiveKit to assign
    a job. The agent publishes this attribute after it has connected and loaded
    all session resources, so the caller token is not returned during startup.
    """
    timeout_s = (
        float(os.getenv("VOICE_AGENT_READY_TIMEOUT_S", str(DEFAULT_AGENT_READY_TIMEOUT_S)))
        if timeout_s is None
        else timeout_s
    )
    poll_interval_s = (
        float(os.getenv("VOICE_AGENT_READY_POLL_S", str(DEFAULT_AGENT_READY_POLL_S)))
        if poll_interval_s is None
        else poll_interval_s
    )
    deadline = asyncio.get_running_loop().time() + timeout_s

    while True:
        response = await lkapi.room.list_participants(
            api.ListParticipantsRequest(room=room_name)
        )
        for participant in response.participants:
            if participant.attributes.get(AGENT_READY_ATTRIBUTE) == "true":
                return participant

        remaining_s = deadline - asyncio.get_running_loop().time()
        if remaining_s <= 0:
            raise TimeoutError(f"agent did not become ready within {timeout_s:.1f}s")
        await asyncio.sleep(min(poll_interval_s, remaining_s))

class VoiceAgentService:
    async def start_call(self):

        session_id = str(uuid.uuid4())
        room_name = f"call-{session_id}"

        lkapi = get_lkapi()

        try:
            await create_room(lkapi, room_name)
            await lkapi.agent_dispatch.create_dispatch(
                api.CreateAgentDispatchRequest(
                    agent_name="local-voice-agent",
                    room=room_name,
                )
            )
            await wait_for_agent_ready(lkapi, room_name)
            token = generate_token(
                room_name=room_name,
                user_id=session_id,
                email="guest@example.com",
            )
            return StartCallResponse(
                session_id=session_id,
                room_name=room_name,
                livekit_url=os.environ["LIVEKIT_URL"],
                token=token,
            )

        finally:
            await lkapi.aclose()
