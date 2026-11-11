"""Kokoro ONNX Runtime adapter for LiveKit Agents."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import numpy as np

from livekit.agents import APIConnectOptions
from livekit.agents.tts import AudioEmitter, ChunkedStream, TTS, TTSCapabilities

from voice_logging import format_event


logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "models/kokoro/kokoro-v1.0.onnx"
DEFAULT_VOICES_PATH = "models/kokoro/voices-v1.0.bin"
DEFAULT_VOICE = "af_heart"
DEFAULT_LANG = "en-us"
DEFAULT_SPEED = 1.0
DEFAULT_SAMPLE_RATE = 24000


@dataclass(frozen=True)
class KokoroONNXFirstAudio:
    synthesis_latency_s: float


def _load_kokoro(model_path: str, voices_path: str):
    """Load Kokoro ONNX and its voice pack on the dedicated worker."""
    from kokoro_onnx import Kokoro

    return Kokoro(model_path, voices_path)


class KokoroONNXTTS(TTS):
    """CPU ONNX Kokoro adapter with blocking inference off the event loop."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        voices_path: str = DEFAULT_VOICES_PATH,
        *,
        voice: str = DEFAULT_VOICE,
        lang: str = DEFAULT_LANG,
        speed: float = DEFAULT_SPEED,
        sentence_pause: float = 0.25,
        clause_pause: float = 0.1,
    ) -> None:
        resolved_model_path = Path(model_path).expanduser()
        resolved_voices_path = Path(voices_path).expanduser()
        for label, path in (
            ("Kokoro ONNX model", resolved_model_path),
            ("Kokoro ONNX voices", resolved_voices_path),
        ):
            if not path.is_file():
                raise FileNotFoundError(
                    f"{label} not found: {path}. Set the corresponding "
                    "KOKORO_ONNX_*_PATH environment variable."
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
        self._voices_path = str(resolved_voices_path)
        self._voice = voice
        self._lang = lang
        self._speed = speed
        self._sentence_pause = sentence_pause
        self._clause_pause = clause_pause
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="kokoro-onnx-cpu",
        )
        self._kokoro_future = self._executor.submit(
            _load_kokoro,
            self._model_path,
            self._voices_path,
        )

    @property
    def model(self) -> str:
        return self._model_path

    @property
    def provider(self) -> str:
        return "local-kokoro-onnx"

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions | None = None,
    ) -> ChunkedStream:
        if conn_options is None:
            conn_options = APIConnectOptions()
        return KokoroONNXChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
        )

    def prewarm(self) -> None:
        # Load and execute one short utterance so ONNX graph/session setup is
        # paid during process prewarm instead of on the first caller turn.
        kokoro = self._kokoro_future.result()
        kokoro.create(
            "Hello.",
            voice=self._voice,
            speed=self._speed,
            lang=self._lang,
            sentence_pause=self._sentence_pause,
            clause_pause=self._clause_pause,
        )

    async def aclose(self) -> None:
        await asyncio.to_thread(self._executor.shutdown, True, cancel_futures=True)


class KokoroONNXChunkedStream(ChunkedStream):
    def __init__(self, *, tts: KokoroONNXTTS, input_text: str, conn_options):
        super().__init__(
            tts=tts,
            input_text=input_text,
            conn_options=conn_options,
        )
        self._kokoro = tts

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
                "KOKORO ONNX REQUEST",
                request=request_id,
                chars=len(self._input_text),
            )
        )

        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._kokoro.sample_rate,
            num_channels=self._kokoro.num_channels,
            mime_type="audio/pcm",
            stream=False,
        )

        def generate_audio() -> None:
            try:
                worker_started_at = time.perf_counter()
                logger.info(
                    format_event(
                        "KOKORO ONNX WORKER START",
                        request=request_id,
                        queue_wait_s=worker_started_at - synthesis_requested_at,
                    )
                )
                kokoro = self._kokoro._kokoro_future.result()
                audio, sample_rate = kokoro.create(
                    self._input_text,
                    voice=self._kokoro._voice,
                    speed=self._kokoro._speed,
                    lang=self._kokoro._lang,
                    sentence_pause=self._kokoro._sentence_pause,
                    clause_pause=self._kokoro._clause_pause,
                )
                pcm_audio = (
                    np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
                    * 32767
                ).astype(np.int16)
                if int(sample_rate) != self._kokoro.sample_rate:
                    logger.warning(
                        format_event(
                            "KOKORO ONNX SAMPLE RATE MISMATCH",
                            expected=self._kokoro.sample_rate,
                            actual=sample_rate,
                        )
                    )
                if not cancelled.is_set():
                    loop.call_soon_threadsafe(queue.put_nowait, pcm_audio.tobytes())
            except BaseException as exc:
                logger.exception(
                    format_event(
                        "KOKORO ONNX ERROR",
                        request=request_id,
                        error=type(exc).__name__,
                    )
                )
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        loop.run_in_executor(self._kokoro._executor, generate_audio)

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
                            "KOKORO ONNX FIRST PCM",
                            request=request_id,
                            elapsed_s=latency,
                        )
                    )
                    self._kokoro.emit(
                        "first_audio",  # type: ignore[arg-type]
                        KokoroONNXFirstAudio(synthesis_latency_s=latency),
                    )

                output_emitter.push(chunk)
        finally:
            cancelled.set()
            output_emitter.flush()
            logger.info(
                format_event(
                    "KOKORO ONNX COMPLETE",
                    request=request_id,
                    elapsed_s=time.perf_counter() - synthesis_requested_at,
                    cancelled=not completed,
                )
            )
