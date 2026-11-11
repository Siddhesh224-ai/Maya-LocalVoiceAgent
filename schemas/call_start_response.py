from pydantic import BaseModel


class StartCallResponse(BaseModel):
    session_id: str
    room_name: str
    livekit_url: str
    token: str
