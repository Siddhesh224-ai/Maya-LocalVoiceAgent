
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class StreamText:
    text: str
    first_token_latency_s: float | None = None


class PunctuationChunker:
    """Emit text as soon as a short spoken phrase reaches punctuation."""

    def __init__(self, boundaries: Iterable[str] = (",", ".", "?", "!", ";")) -> None:
        self._boundaries = tuple(boundaries)
        self._buffer = ""

    def push(self, text: str) -> list[str]:
        self._buffer += text
        chunks: list[str] = []
        start = 0
        for index, char in enumerate(self._buffer):
            if char in self._boundaries:
                chunk = self._buffer[start : index + 1].strip()
                if chunk:
                    chunks.append(chunk)
                start = index + 1
        self._buffer = self._buffer[start:]
        return chunks

    def flush(self) -> str | None:
        chunk = self._buffer.strip()
        self._buffer = ""
        return chunk or None


class LocalLLMClient:
    """Streaming client for llama-server/Qwen Chat Completions."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080/v1",
        model: str = "Qwen2.5-3B-Instruct-4bit",
        *,
        connect_timeout_s: float = 0.2,
        read_timeout_s: float = 120.0,
    ) -> None:
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=connect_timeout_s,
                read=read_timeout_s,
                write=read_timeout_s,
                pool=connect_timeout_s,
            ),
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
            http2=False,
        )

    async def stream_text(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 128,
    ) -> AsyncIterator[StreamText]:
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        chunker = PunctuationChunker()
        started = time.perf_counter()
        first_token_at: float | None = None

        try:
            async with self._client.stream("POST", self._url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = event.get("choices", [{}])[0].get("delta", {})
                    text = delta.get("content") or ""
                    if not text:
                        continue
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    for phrase in chunker.push(text):
                        yield StreamText(
                            text=phrase,
                            first_token_latency_s=first_token_at - started,
                        )
                tail = chunker.flush()
                if tail:
                    yield StreamText(
                        text=tail,
                        first_token_latency_s=(first_token_at - started if first_token_at else None),
                    )
        except asyncio.CancelledError:
            # Exiting the async context closes the response/socket immediately.
            raise

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "LocalLLMClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()
