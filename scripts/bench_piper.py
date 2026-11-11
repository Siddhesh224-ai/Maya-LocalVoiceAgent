#!/usr/bin/env python3
"""Measure steady-state Piper first-audio latency and real-time factor."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from piper import PiperVoice


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="models/piper/en_US-lessac-low.onnx",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--text",
        default="I can help with that question and give you a concise answer right now.",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path).expanduser()
    if not model_path.is_file():
        parser.error(
            f"Piper model not found: {model_path}. Download it first with "
            "`python -m piper.download_voices --download-dir models/piper "
            "en_US-lessac-low`."
        )
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    load_started = time.perf_counter()
    voice = PiperVoice.load(str(model_path), use_cuda=False)
    load_elapsed = time.perf_counter() - load_started

    # Exclude model loading and first-call ONNX setup from steady-state numbers.
    for _ in voice.synthesize("Warm up the local speech engine."):
        pass

    first_audio: list[float] = []
    total_elapsed: list[float] = []
    rtfs: list[float] = []
    audio_seconds = 0.0

    for _ in range(args.repeats):
        started = time.perf_counter()
        first = None
        samples = 0
        for chunk in voice.synthesize(args.text):
            if first is None:
                first = time.perf_counter() - started
            samples += len(chunk.audio_float_array)

        elapsed = time.perf_counter() - started
        duration = samples / voice.config.sample_rate
        audio_seconds = duration
        first_audio.append(first if first is not None else elapsed)
        total_elapsed.append(elapsed)
        rtfs.append(elapsed / duration if duration else float("inf"))

    print(f"model={model_path}")
    print(f"sample_rate={voice.config.sample_rate}")
    print(f"load_s={load_elapsed:.3f}")
    print(f"audio_duration_s={audio_seconds:.3f}")
    print(f"first_audio_p50_s={statistics.median(first_audio):.3f}")
    print(f"total_synthesis_p50_s={statistics.median(total_elapsed):.3f}")
    print(f"rtf_p50={statistics.median(rtfs):.3f}")
    print(f"first_audio_samples={[round(value, 3) for value in first_audio]}")
    print(f"rtf_samples={[round(value, 3) for value in rtfs]}")


if __name__ == "__main__":
    main()
