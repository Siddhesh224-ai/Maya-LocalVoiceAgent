
from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Optional

from livekit.agents import APIConnectOptions
from livekit.agents.tts import AudioEmitter, ChunkedStream, TTS, TTSCapabilities

from voice_logging import format_event


logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "models/piper/en_US-lessac-low.onnx"
DEFAULT_SAMPLE_RATE = 22050


@dataclass(frozen=True)
class PiperFirstAudio:
    synthesis_latency_s: float


def _load_voice(model_path: str):
    """Load Piper lazily on the dedicated CPU worker."""
    from piper import PiperVoice

    return PiperVoice.load(model_path, use_cuda=False)


class PiperTTS(TTS):
    """Piper TTS that keeps blocking ONNX inference off the event loop."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        *,
        length_scale: Optional[float] = None,
        noise_scale: Optional[float] = None,
        noise_w_scale: Optional[float] = None,
    ) -> None:
        resolved_model_path = Path(model_path).expanduser()
        if not resolved_model_path.is_file():
            raise FileNotFoundError(
                f"Piper voice model not found: {resolved_model_path}. "
                "Download a voice with `python -m piper.download_voices "
                "--download-dir models/piper en_US-lessac-low` or set PIPER_MODEL_PATH."
            )

        super().__init__(
            capabilities=TTSCapabilities(
                streaming=False,
                aligned_transcript=False,
            ),
            sample_rate=DEFAULT_SAMPLE_RATE,
            num_channels=1,
        )

        self._model_path = str(resolved_model_path)
        from piper.config import SynthesisConfig

        self._synthesis_config = SynthesisConfig(
            length_scale=length_scale,
            noise_scale=noise_scale,
            noise_w_scale=noise_w_scale,
        )
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="piper-cpu",
        )
        self._voice_future = self._executor.submit(_load_voice, self._model_path)

    @property
    def model(self) -> str:
        return self._model_path

    @property
    def provider(self) -> str:
        return "local-piper"

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions | None = None,
    ) -> ChunkedStream:
        if conn_options is None:
            conn_options = APIConnectOptions()
        return PiperChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
        )

    def prewarm(self) -> None:
        voice = self._voice_future.result()
        self._sample_rate = int(voice.config.sample_rate)

    async def aclose(self) -> None:
        await asyncio.to_thread(self._executor.shutdown, True, cancel_futures=True)


class PiperChunkedStream(ChunkedStream):
    def __init__(self, *, tts: PiperTTS, input_text: str, conn_options):
        super().__init__(
            tts=tts,
            input_text=input_text,
            conn_options=conn_options,
        )
        self._piper = tts

    async def _run(self, output_emitter: AudioEmitter) -> None:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[bytes | BaseException | None] = asyncio.Queue()
        cancelled = Event()
        completed = False
        synthesis_requested_at = time.perf_counter()
        first_audio_emitted = False
        request_id = uuid.uuid4().hex

        logger.info(
            format_event(
                "PIPER REQUEST",
                request=request_id,
                chars=len(self._input_text),
            )
        )

        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._piper.sample_rate,
            num_channels=self._piper.num_channels,
            mime_type="audio/pcm",
            stream=False,
        )

        def generate_audio() -> None:
            try:
                worker_started_at = time.perf_counter()
                logger.info(
                    format_event(
                        "PIPER WORKER START",
                        request=request_id,
                        queue_wait_s=worker_started_at - synthesis_requested_at,
                    )
                )
                voice = self._piper._voice_future.result()
                for result in voice.synthesize(
                    self._input_text,
                    syn_config=self._piper._synthesis_config,
                ):
                    if cancelled.is_set():
                        return
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        result.audio_int16_bytes,
                    )
            except BaseException as exc:
                logger.exception(
                    format_event(
                        "PIPER ERROR",
                        request=request_id,
                        error=type(exc).__name__,
                    )
                )
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        loop.run_in_executor(self._piper._executor, generate_audio)

        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    completed = True
                    break
                if isinstance(chunk, BaseException):
                    raise chunk

                if not first_audio_emitted:
                    first_audio_emitted = True
                    latency = time.perf_counter() - synthesis_requested_at
                    logger.info(
                        format_event(
                            "PIPER FIRST PCM",
                            request=request_id,
                            elapsed_s=latency,
                        )
                    )
                    self._piper.emit(
                        "first_audio",  # type: ignore[arg-type]
                        PiperFirstAudio(synthesis_latency_s=latency),
                    )

                output_emitter.push(chunk)
        finally:
            cancelled.set()
            output_emitter.flush()
            logger.info(
                format_event(
                    "PIPER COMPLETE",
                    request=request_id,
                    elapsed_s=time.perf_counter() - synthesis_requested_at,
                    cancelled=not completed,
                )
            )
