from fastapi import APIRouter, Depends, HTTPException

from schemas.call_start_response import StartCallResponse
from services.voice_agent_service import VoiceAgentService


router = APIRouter(prefix="/api/calls", tags=["Calls"])


@router.post("/start", response_model=StartCallResponse)
async def start_call(
    service: VoiceAgentService = Depends(VoiceAgentService),
):
    try:
        return await service.start_call()
    except TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail="The voice agent did not become ready in time. Please retry.",
        ) from exc
