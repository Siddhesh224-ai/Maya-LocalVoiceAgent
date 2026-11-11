from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.voice_agent_router import router as voice_agent_router

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(voice_agent_router)


@app.get("/health", tags=["System"])
async def health() -> dict[str, str]:
    """Return a lightweight liveness response for local smoke tests."""
    return {"status": "ok"}
