import asyncio
import types
import unittest
from unittest.mock import patch

from agent import create_local_tts
from piper_tts import PiperTTS


class PiperTTSTest(unittest.TestCase):
    def test_agent_selects_piper_from_environment(self) -> None:
        with (
            patch("agent.PiperTTS") as piper,
            patch.dict(
                "os.environ",
                {
                    "VOICE_TTS_BACKEND": "piper",
                    "PIPER_MODEL_PATH": "models/piper/test.onnx",
                },
                clear=False,
            ),
        ):
            create_local_tts()

        piper.assert_called_once_with(model_path="models/piper/test.onnx")

    def test_agent_passes_optional_piper_prosody_controls(self) -> None:
        with (
            patch("agent.PiperTTS") as piper,
            patch.dict(
                "os.environ",
                {
                    "VOICE_TTS_BACKEND": "piper",
                    "PIPER_MODEL_PATH": "models/piper/test.onnx",
                    "PIPER_LENGTH_SCALE": "0.98",
                    "PIPER_NOISE_SCALE": "0.75",
                    "PIPER_NOISE_W_SCALE": "0.85",
                },
                clear=False,
            ),
        ):
            create_local_tts()

        piper.assert_called_once_with(
            model_path="models/piper/test.onnx",
            length_scale=0.98,
            noise_scale=0.75,
            noise_w_scale=0.85,
        )

    def test_piper_loads_voice_on_its_dedicated_worker(self) -> None:
        initialized_on: list[int] = []

        def load(_model_path: str) -> object:
            import threading

            initialized_on.append(threading.get_ident())
            return types.SimpleNamespace(config=types.SimpleNamespace(sample_rate=22050))

        with (
            patch("piper_tts.Path.is_file", return_value=True),
            patch("piper_tts._load_voice", side_effect=load),
        ):
            tts = PiperTTS(model_path="test.onnx")
            try:
                tts.prewarm()
                self.assertEqual(tts.sample_rate, 22050)
            finally:
                asyncio.run(tts.aclose())

        import threading

        self.assertEqual(len(initialized_on), 1)
        self.assertNotEqual(initialized_on[0], threading.get_ident())
