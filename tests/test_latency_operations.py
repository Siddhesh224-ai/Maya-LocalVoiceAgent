import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent import _pop_completed_sentence, _prewarm_process
from livekit.agents.metrics import EOUMetrics, TTSMetrics
from agent import TurnLatencyTable
from voice_logging import configure_logging, format_event, patch_livekit_cli_logging


class FirstSpeechSegmentationTest(unittest.TestCase):
    def test_flushes_a_short_natural_first_sentence(self) -> None:
        sentence, remainder = _pop_completed_sentence("Yes, I can. I will check that now.")

        self.assertEqual(sentence, "Yes, I can.")
        self.assertEqual(remainder, "I will check that now.")


class ProcessPrewarmTest(unittest.TestCase):
    def test_process_prewarm_stores_models_for_the_job(self) -> None:
        proc = SimpleNamespace(userdata={})
        stt = SimpleNamespace(prewarm=lambda: None)
        tts = SimpleNamespace(prewarm=lambda: None)

        with (
            patch("agent.silero", None),
            patch("agent._rag_enabled", return_value=False),
            patch("agent.MoonshineStreamingSTT", return_value=stt),
            patch("agent.PiperTTS", return_value=tts),
            patch.dict("os.environ", {"VOICE_TTS_BACKEND": "piper"}, clear=False),
        ):
            _prewarm_process(proc)

        self.assertIs(proc.userdata["stt"], stt)
        self.assertIs(proc.userdata["tts"], tts)
        self.assertTrue(proc.userdata["local_models_prewarmed"])


class ReadableLoggingTest(unittest.TestCase):
    def test_format_event_is_human_readable_and_stable(self) -> None:
        self.assertEqual(
            format_event("TURN COMPLETE", turn="turn-1", elapsed_s=1.234, error=None),
            "TURN COMPLETE | turn=turn-1 | elapsed_s=1.234 | error=none",
        )

    def test_livekit_cli_logging_patch_keeps_output_file_only(self) -> None:
        configure_logging()
        patch_livekit_cli_logging()

        from livekit.agents.cli import _legacy
        from livekit.agents.cli import cli as cli_module

        _legacy._configure_logger(None, "INFO")
        cli_module.setup_logging("INFO", devmode=True, console=True)

        root_handlers = __import__("logging").getLogger().handlers
        self.assertTrue(root_handlers)
        self.assertTrue(all(handler.__class__.__name__ == "RotatingFileHandler" for handler in root_handlers))


class WallClockLatencyTest(unittest.TestCase):
    def test_records_first_audio_from_component_timestamps(self) -> None:
        table = TurnLatencyTable()
        eou = EOUMetrics(
            timestamp=0.0,
            end_of_utterance_delay=0.4,
            transcription_delay=0.4,
            on_user_turn_completed_delay=0.0,
            speech_id="turn-1",
        )
        tts = TTSMetrics(
            label="kokoro",
            request_id="request-1",
            timestamp=0.0,
            ttfb=0.2,
            duration=0.5,
            audio_duration=1.0,
            cancelled=False,
            characters_count=20,
            streamed=False,
            speech_id="turn-1",
        )

        with patch("agent.time.perf_counter", side_effect=[100.0, 104.0]):
            table.record_metric(eou)
            table.record_metric(tts)

        self.assertAlmostEqual(table._turns["turn-1"]["wall_first_audio_s"], 4.1)

