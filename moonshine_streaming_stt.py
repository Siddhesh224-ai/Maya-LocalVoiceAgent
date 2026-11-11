
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
from livekit import rtc
from livekit.agents import stt
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from moonshine_voice import ModelArch, TranscriptEventListener, Transcriber, get_model_for_language


class _TranscriptAccumulator:
    """Keep completed Moonshine lines plus the currently changing line."""

    def __init__(self) -> None:
        self._final_lines: list[str] = []
        self._current_line = ""

    @property
    def full(self) -> str:
        return " ".join(line for line in (*self._final_lines, self._current_line) if line).strip()

    def interim(self, text: str) -> str:
        self._current_line = text.strip()
        return self.full

    def complete(self, text: str) -> tuple[str, str]:
        line = text.strip()
        if line:
            self._final_lines.append(line)
        self._current_line = ""
        return line, self.full

    def reset(self) -> None:
        self._final_lines.clear()
        self._current_line = ""


class MoonshineStreamingSTT(stt.STT):
    """Run Moonshine's local streaming transcriber behind LiveKit's STT API."""

    def __init__(
        self,
        *,
        model_dir: str | Path = "models/moonshine",
        language: str = "en",
        model_arch: str = "medium-streaming",
        update_interval: float = 0.5,
        options: dict[str, object] | None = None,
        transcriber_factory: Callable[..., object] | None = None,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                manual_flush=True,
            )
        )
        self._model_dir = Path(model_dir)
        self._language = language
        self._model_arch = _model_arch_from_name(model_arch)
        self._update_interval = update_interval
        self._options = options or {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="moonshine-stt")
        self._transcriber_future = self._executor.submit(
            transcriber_factory or _create_transcriber,
            self._model_dir,
            self._language,
            self._model_arch,
            self._update_interval,
            self._options,
        )

    @property
    def model(self) -> str:
        return self._model_arch.name.lower().replace("_", "-")

    @property
    def provider(self) -> str:
        return "moonshine-voice"

    def prewarm(self) -> None:
        """Block until the local model is available and loaded."""
        self._transcriber_future.result()

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.RecognizeStream:
        stream_language = language if language is not NOT_GIVEN else self._language
        return _MoonshineRecognizeStream(
            stt=self,
            language=str(stream_language),
            conn_options=conn_options,
        )

    async def aclose(self) -> None:
        def close() -> None:
            transcriber = self._transcriber_future.result()
            transcriber.close()

        try:
            await asyncio.to_thread(close)
        finally:
            self._executor.shutdown(wait=True, cancel_futures=True)

    async def _recognize_impl(self, *args, **kwargs) -> stt.SpeechEvent:
        raise NotImplementedError("MoonshineStreamingSTT is streaming-only; use stream().")


class _MoonshineRecognizeStream(stt.RecognizeStream):
    def __init__(self, *, stt: MoonshineStreamingSTT, language: str, conn_options: APIConnectOptions):
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=16_000)
        self._moonshine_stt = stt
        self._language = language
        self._moonshine_stream: object | None = None
        self._events: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self._audio_duration_s = 0.0
        self._last_interim = ""
        self._last_final = ""
        self._transcript = _TranscriptAccumulator()
        self._audio_batch: list[np.ndarray] = []
        self._audio_batch_samples = 0
        self._speech_started = False
        self._final_event_sent = False

    async def _run(self) -> None:
        async for item in self._input_ch:
            if isinstance(item, self._FlushSentinel):
                await self._on_flush()
                continue

            self._audio_duration_s += item.duration
            await self._add_audio(item)
            self._emit_pending_events(final_only=False)

    async def _add_audio(self, frame: rtc.AudioFrame) -> None:
        loop = asyncio.get_running_loop()
        if self._moonshine_stream is None:
            self._moonshine_stream = await loop.run_in_executor(
                self._moonshine_stt._executor,
                self._new_moonshine_stream,
                loop,
            )
        samples = _frame_to_mono_samples(frame)
        if frame.sample_rate != 16_000:
            raise ValueError(f"Moonshine requires 16 kHz audio, received {frame.sample_rate}")
        self._audio_batch.append(samples)
        self._audio_batch_samples += len(samples)
        if self._audio_batch_samples >= int(frame.sample_rate * 0.05):
            await self._flush_audio_batch(frame.sample_rate)

    async def _flush_audio_batch(self, sample_rate: int = 16_000) -> None:
        if not self._audio_batch or self._moonshine_stream is None:
            return
        loop = asyncio.get_running_loop()
        samples = (
            self._audio_batch[0]
            if len(self._audio_batch) == 1
            else np.concatenate(self._audio_batch)
        )
        self._audio_batch.clear()
        self._audio_batch_samples = 0
        # Moonshine's Python wrapper accepts any float sequence and creates the
        # C float buffer itself, so passing float32 NumPy data avoids .tolist().
        await loop.run_in_executor(
            self._moonshine_stt._executor,
            self._moonshine_stream.add_audio,
            samples,
            sample_rate,
        )

    def _new_moonshine_stream(self, loop: asyncio.AbstractEventLoop) -> object:
        transcriber = self._moonshine_stt._transcriber_future.result()
        stream = transcriber.create_stream(update_interval=self._moonshine_stt._update_interval)
        stream.add_listener(_MoonshineListener(loop, self._events))
        stream.start()
        return stream

    async def _on_flush(self) -> None:
        suppress_stop_finals = self._final_event_sent
        if self._moonshine_stream is not None:
            loop = asyncio.get_running_loop()
            await self._flush_audio_batch()
            await asyncio.sleep(0)
            self._emit_pending_events(final_only=False)
            # LiveKit may commit a turn before Moonshine publishes its
            # definitive line from stream.stop(). Commit the unfinished line
            # before stop() so the final transcript is available in time.
            suppress_stop_finals = self._final_event_sent
            if self._transcript._current_line:
                self._event_ch.send_nowait(
                    self._event(
                        stt.SpeechEventType.FINAL_TRANSCRIPT,
                        self._transcript._current_line,
                    )
                )
                self._final_event_sent = True
                suppress_stop_finals = True
            await loop.run_in_executor(self._moonshine_stt._executor, self._moonshine_stream.stop)
            await asyncio.sleep(0)
        self._emit_pending_events(final_only=True, emit_finals=not suppress_stop_finals)
        if not self._final_event_sent and self._transcript.full:
            self._event_ch.send_nowait(
                self._event(stt.SpeechEventType.FINAL_TRANSCRIPT, self._transcript.full)
            )
            self._final_event_sent = True
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.RECOGNITION_USAGE,
                recognition_usage=stt.RecognitionUsage(audio_duration=self._audio_duration_s),
            )
        )
        await self._close_segment()
        self._reset_segment()

    def _emit_pending_events(self, *, final_only: bool, emit_finals: bool = True) -> None:
        while not self._events.empty():
            event_type, text = self._events.get_nowait()
            text = text.strip()
            if not text:
                continue
            if not self._speech_started:
                self._speech_started = True
                self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))
            if event_type == "final":
                line, full = self._transcript.complete(text)
                self._last_final = full
                self._last_interim = full
                # LiveKit concatenates FINAL_TRANSCRIPT alternatives into the
                # turn transcript. During the stop() drain, suppress finals
                # when the unfinished line was already committed above.
                if emit_finals:
                    self._event_ch.send_nowait(
                        self._event(stt.SpeechEventType.FINAL_TRANSCRIPT, line)
                    )
                    self._final_event_sent = True
                continue
            if not final_only and text != self._last_interim:
                full = self._transcript.interim(text)
                self._last_interim = full
                self._event_ch.send_nowait(
                    self._event(stt.SpeechEventType.INTERIM_TRANSCRIPT, full)
                )

    async def _close_segment(self) -> None:
        if self._moonshine_stream is None:
            return
        stream = self._moonshine_stream
        self._moonshine_stream = None
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._moonshine_stt._executor, stream.close)

    def _reset_segment(self) -> None:
        self._audio_duration_s = 0.0
        self._last_interim = ""
        self._last_final = ""
        self._transcript.reset()
        self._audio_batch.clear()
        self._audio_batch_samples = 0
        self._speech_started = False
        self._final_event_sent = False

    def _event(self, event_type: stt.SpeechEventType, text: str) -> stt.SpeechEvent:
        return stt.SpeechEvent(
            type=event_type,
            alternatives=[stt.SpeechData(text=text, language=self._language)],
        )


class _MoonshineListener(TranscriptEventListener):
    """Move Moonshine's worker-thread callbacks onto the LiveKit event loop."""

    def __init__(self, loop: asyncio.AbstractEventLoop, events: asyncio.Queue[tuple[str, str]]) -> None:
        self._loop = loop
        self._events = events

    def on_line_text_changed(self, event) -> None:
        self._loop.call_soon_threadsafe(self._events.put_nowait, ("interim", event.line.text))

    def on_line_completed(self, event) -> None:
        self._loop.call_soon_threadsafe(self._events.put_nowait, ("final", event.line.text))


def _model_arch_from_name(model_arch: str) -> ModelArch:
    normalized = model_arch.strip().upper().replace("-", "_")
    try:
        return ModelArch[normalized]
    except KeyError as exc:
        available = ", ".join(arch.name.lower().replace("_", "-") for arch in ModelArch)
        raise ValueError(f"Unsupported Moonshine model architecture {model_arch!r}; choose one of: {available}") from exc


def _create_transcriber(
    model_dir: Path,
    language: str,
    model_arch: ModelArch,
    update_interval: float,
    options: dict[str, object],
) -> Transcriber:
    model_path, resolved_arch = get_model_for_language(
        language,
        model_arch,
        cache_root=model_dir,
    )
    return Transcriber(
        model_path=model_path,
        model_arch=resolved_arch,
        update_interval=update_interval,
        options=options,
    )


def _frame_to_mono_samples(frame: rtc.AudioFrame) -> np.ndarray:
    samples = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32)
    if frame.num_channels > 1:
        samples = samples.reshape(-1, frame.num_channels).mean(axis=1)
    return samples / 32768.0
