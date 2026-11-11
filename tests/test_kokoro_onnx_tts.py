import asyncio
import types
import unittest
from unittest.mock import patch

from agent import create_local_tts
from kokoro_onnx_tts import KokoroONNXTTS


class KokoroONNXTtsTest(unittest.TestCase):
    def test_mlx_kokoro_backend_is_no_longer_active(self) -> None:
        with patch.dict("os.environ", {"VOICE_TTS_BACKEND": "kokoro"}, clear=False):
            with self.assertRaisesRegex(ValueError, "piper.*kokoro-onnx"):
                create_local_tts()

    def test_agent_selects_kokoro_onnx_from_environment(self) -> None:
        with (
            patch("agent.KokoroONNXTTS") as kokoro,
            patch.dict(
                "os.environ",
                {
                    "VOICE_TTS_BACKEND": "kokoro-onnx",
                    "KOKORO_ONNX_MODEL_PATH": "models/kokoro/kokoro-v1.0.onnx",
                    "KOKORO_ONNX_VOICES_PATH": "models/kokoro/voices-v1.0.bin",
                    "KOKORO_ONNX_VOICE": "af_heart",
                },
                clear=False,
            ),
        ):
            create_local_tts()

        kokoro.assert_called_once_with(
            model_path="models/kokoro/kokoro-v1.0.onnx",
            voices_path="models/kokoro/voices-v1.0.bin",
            voice="af_heart",
            lang="en-us",
            speed=1.0,
            sentence_pause=0.25,
            clause_pause=0.1,
        )

    def test_model_loads_on_dedicated_worker(self) -> None:
        initialized_on: list[int] = []

        def load(_model_path: str, _voices_path: str) -> object:
            import threading

            initialized_on.append(threading.get_ident())
            return types.SimpleNamespace(create=lambda *_args, **_kwargs: (None, 24000))

        with (
            patch("kokoro_onnx_tts.Path.is_file", return_value=True),
            patch("kokoro_onnx_tts._load_kokoro", side_effect=load),
        ):
            tts = KokoroONNXTTS(
                model_path="test.onnx",
                voices_path="test.bin",
                voice="af_heart",
            )
            try:
                tts.prewarm()
                self.assertEqual(tts.sample_rate, 24000)
            finally:
                asyncio.run(tts.aclose())

        import threading

        self.assertEqual(len(initialized_on), 1)
        self.assertNotEqual(initialized_on[0], threading.get_ident())
