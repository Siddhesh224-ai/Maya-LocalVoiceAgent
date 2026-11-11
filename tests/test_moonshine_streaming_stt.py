import asyncio
import unittest

import numpy as np
from livekit import rtc

from moonshine_streaming_stt import MoonshineStreamingSTT


class _FakeLine:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeEvent:
    def __init__(self, text: str) -> None:
        self.line = _FakeLine(text)


class _FakeMoonshineStream:
    def __init__(self) -> None:
        self.listener = None

    def add_listener(self, listener) -> None:
        self.listener = listener

    def start(self) -> None:
        pass

    def add_audio(self, _samples, _sample_rate: int) -> None:
        self.listener.on_line_text_changed(_FakeEvent("hello moonshine"))

    def stop(self) -> None:
        self.listener.on_line_completed(_FakeEvent("hello moonshine"))

    def close(self) -> None:
        pass


class _FakeMoonshineTranscriber:
    def create_stream(self, **_kwargs) -> _FakeMoonshineStream:
        return _FakeMoonshineStream()

    def close(self) -> None:
        pass


class MoonshineStreamingSTTTest(unittest.TestCase):
    def test_stream_emits_interim_and_final_transcripts(self) -> None:
        async def exercise() -> list[str]:
            stt = MoonshineStreamingSTT(
                model_dir="unused",
                transcriber_factory=lambda *_: _FakeMoonshineTranscriber(),
            )
            try:
                stream = stt.stream()
                stream.push_frame(
                    rtc.AudioFrame(
                        data=np.zeros(320, dtype=np.int16).tobytes(),
                        sample_rate=16_000,
                        num_channels=1,
                        samples_per_channel=320,
                    )
                )
                await asyncio.sleep(0.02)
                stream.flush()
                events = []
                while len(events) < 4:
                    events.append(await asyncio.wait_for(stream.__anext__(), timeout=1))
                    if events[-1].type.value == "recognition_usage":
                        break
                await stream.aclose()
                event_types = [event.type.value for event in events]
                self.assertEqual(event_types.count("final_transcript"), 1)
                return event_types
            finally:
                await stt.aclose()

        event_types = asyncio.run(exercise())

        self.assertIn("interim_transcript", event_types)
        self.assertIn("final_transcript", event_types)
        self.assertIn("recognition_usage", event_types)
