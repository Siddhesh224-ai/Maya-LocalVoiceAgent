import asyncio
import os
import logging
import resource
import time
import uuid
import re
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

os.environ.setdefault("ORT_LOGGING_LEVEL", "3")

from livekit import agents
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    APIConnectOptions,
    TurnHandlingOptions,
    inference,
    llm,
    FlushSentinel,
)
from livekit.agents.voice.agent_session import SessionConnectOptions
from dotenv import load_dotenv

from rag import ensure_rag_ready, retrieve_with_timing
from moonshine_streaming_stt import MoonshineStreamingSTT
from kokoro_onnx_tts import KokoroONNXTTS
from piper_tts import PiperTTS
from voice_logging import configure_logging, format_event, patch_livekit_cli_logging, set_log_context
from livekit.plugins import openai
try:
    from livekit.plugins import silero
except ImportError:  # pragma: no cover - optional local plugin installation
    silero = None
from livekit.agents.metrics import EOUMetrics, InterruptionMetrics, LLMMetrics, TTSMetrics

load_dotenv()
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
configure_logging()
patch_livekit_cli_logging()

logger = logging.getLogger(__name__)

AGENT_READY_ATTRIBUTE = "voice_agent.ready"

server = AgentServer(
    num_idle_processes=int(os.getenv("VOICE_IDLE_PROCESSES", "1")),
    job_memory_warn_mb=float(os.getenv("VOICE_JOB_MEMORY_WARN_MB", "2500")),
    job_memory_limit_mb=float(os.getenv("VOICE_JOB_MEMORY_LIMIT_MB", "0")),
    initialize_process_timeout=float(os.getenv("VOICE_PROCESS_INIT_TIMEOUT", "120")),
)


class VoiceTrace:
    """Wall-clock step trace for one speech turn.

    ``perf_counter`` is intentionally used for durations because it is monotonic.
    The trace stores only step names and timings, never transcript contents.
    """

    def __init__(self, turn_id: str | None = None, *, clock=time.perf_counter) -> None:
        self.turn_id = turn_id or uuid.uuid4().hex[:12]
        self._clock = clock
        self.marks: dict[str, float] = {}
        self.steps: list[str] = []

    def mark(self, step: str) -> None:
        self.mark_at(step, self._clock())

    def mark_at(self, step: str, timestamp: float) -> None:
        if step not in self.marks:
            self.marks[step] = timestamp
            self.steps.append(step)

    def elapsed(self, start: str, end: str) -> float | None:
        if start not in self.marks or end not in self.marks:
            return None
        return self.marks[end] - self.marks[start]

    def snapshot(self) -> dict[str, object]:
        durations = {
            f"{start}_to_{end}": self.marks[end] - self.marks[start]
            for start, end in zip(self.steps, self.steps[1:])
        }
        return {
            "turn": self.turn_id,
            "steps": list(self.steps),
            "marks": dict(self.marks),
            "durations": durations,
        }


def _rag_enabled() -> bool:
    return os.getenv("RAG_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}


def _prewarm_process(proc) -> None:
    """Load every process-shared model before the process receives a job."""
    started_at = time.perf_counter()
    logger.info(format_event("PREWARM START"))
    try:
        if silero is None:
            # The installed native fallback is CPU-only and has no
            # ``force_cpu`` argument; its implementation is already local.
            proc.userdata["vad"] = inference.VAD(
                min_speech_duration=0.05,
                min_silence_duration=0.3,
                activation_threshold=0.5,
            )
        else:
            vad_started_at = time.perf_counter()
            proc.userdata["vad"] = silero.VAD.load(
                min_speech_duration=0.05,
                min_silence_duration=0.3,
                activation_threshold=0.5,
                force_cpu=True,
            )
            logger.info(
                format_event(
                    "PREWARM COMPLETE",
                    component="vad",
                    elapsed_s=time.perf_counter() - vad_started_at,
                )
            )

        detector_started_at = time.perf_counter()
        # v1-mini is local CPU inference in the installed Agents package; keep
        # the object process-shared so session.start does not construct it.
        proc.userdata["turn_detector"] = inference.TurnDetector(version="v1-mini")
        logger.info(
            format_event(
                "PREWARM COMPLETE",
                component="turn_detector",
                elapsed_s=time.perf_counter() - detector_started_at,
            )
        )

        if _rag_enabled():
            rag_started_at = time.perf_counter()
            ensure_rag_ready()
            logger.info(
                format_event(
                    "PREWARM COMPLETE",
                    component="rag",
                    elapsed_s=time.perf_counter() - rag_started_at,
                )
            )

        stt_started_at = time.perf_counter()
        stt = MoonshineStreamingSTT(
            model_dir=os.getenv("MOONSHINE_MODEL_DIR", "models/moonshine"),
            language=os.getenv("MOONSHINE_LANGUAGE", "en"),
            model_arch=os.getenv("MOONSHINE_MODEL_ARCH", "medium-streaming"),
            update_interval=float(os.getenv("MOONSHINE_UPDATE_INTERVAL", "0.1")),
        )
        stt.prewarm()
        logger.info(
            format_event(
                "PREWARM COMPLETE",
                component="moonshine_stt",
                elapsed_s=time.perf_counter() - stt_started_at,
            )
        )

        tts_started_at = time.perf_counter()
        tts = create_local_tts()
        tts.prewarm()
        logger.info(
            format_event(
                "PREWARM COMPLETE",
                component=f"tts:{type(tts).__name__}",
                elapsed_s=time.perf_counter() - tts_started_at,
            )
        )

        # Keep the blocking HTTP warmup off the process setup caller's thread;
        # process prewarm is outside the session event loop.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm-warmup") as executor:
            executor.submit(_warm_local_llm_sync).result()

        proc.userdata.update(
            stt=stt,
            tts=tts,
            local_models_prewarmed=True,
        )
        logger.info(
            format_event(
                "PREWARM READY",
                elapsed_s=time.perf_counter() - started_at,
                peak_rss_mb=_peak_rss_mb(),
            )
        )
    except BaseException:
        logger.exception(format_event("PREWARM ERROR", elapsed_s=time.perf_counter() - started_at))
        raise


def _peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if os.sys.platform == "darwin" else rss / 1024


server.setup_fnc = _prewarm_process

_init_lock = asyncio.Lock()
_init_task: asyncio.Task | None = None
_init_error: BaseException | None = None
_resources: dict[str, object] = {}
is_agent_ready = False


async def _initialize_local_resources(proc=None) -> None:
    global is_agent_ready, _init_error
    try:
        prewarmed = getattr(proc, "userdata", {}) if proc is not None else {}
        if prewarmed.get("local_models_prewarmed"):
            stt = prewarmed["stt"]
            tts = prewarmed["tts"]
            logger.info(format_event("RESOURCES REUSED", source="process_prewarm"))
        else:
            logger.warning(format_event("RESOURCES COLD PATH", reason="process_prewarm_unavailable"))
            if _rag_enabled():
                await asyncio.to_thread(ensure_rag_ready)
            stt = MoonshineStreamingSTT(
                model_dir=os.getenv("MOONSHINE_MODEL_DIR", "models/moonshine"),
                language=os.getenv("MOONSHINE_LANGUAGE", "en"),
                model_arch=os.getenv("MOONSHINE_MODEL_ARCH", "medium-streaming"),
                update_interval=float(os.getenv("MOONSHINE_UPDATE_INTERVAL", "0.1")),
            )
            await asyncio.to_thread(stt.prewarm)
            tts = create_local_tts()
            await asyncio.to_thread(tts.prewarm)
            if proc is not None and "vad" not in getattr(proc, "userdata", {}):
                if silero is not None:
                    proc.userdata["vad"] = await asyncio.to_thread(
                        silero.VAD.load,
                        min_speech_duration=0.05,
                        min_silence_duration=0.3,
                        activation_threshold=0.5,
                        force_cpu=True,
                    )
                else:
                    proc.userdata["vad"] = inference.VAD(
                        min_speech_duration=0.05,
                        min_silence_duration=0.3,
                        activation_threshold=0.5,
                    )
            if proc is not None and "turn_detector" not in getattr(proc, "userdata", {}):
                proc.userdata["turn_detector"] = inference.TurnDetector(version="v1-mini")
            await asyncio.to_thread(_warm_local_llm_sync)
        _resources.update(stt=stt, tts=tts)
        is_agent_ready = True
        logger.info(format_event("RESOURCES READY", peak_rss_mb=_peak_rss_mb()))
    except BaseException as exc:
        _init_error = exc
        logger.exception(format_event("RESOURCE INITIALIZATION ERROR", error=type(exc).__name__))
        raise


async def _ensure_local_resources(proc=None) -> dict[str, object]:
    global _init_task
    async with _init_lock:
        if _init_task is None:
            _init_task = asyncio.create_task(_initialize_local_resources(proc))
    await _init_task
    if _init_error is not None:
        raise RuntimeError("local voice resources failed to initialize") from _init_error
    return _resources


def create_local_tts() -> object:
    """Create the configured local TTS provider."""
    backend = os.getenv("VOICE_TTS_BACKEND", "piper").strip().lower()
    if backend == "piper":
        piper_args: dict[str, object] = {
            "model_path": os.getenv(
                "PIPER_MODEL_PATH",
                "models/piper/en_US-lessac-low.onnx",
            )
        }
        for env_name, argument_name in (
            ("PIPER_LENGTH_SCALE", "length_scale"),
            ("PIPER_NOISE_SCALE", "noise_scale"),
            ("PIPER_NOISE_W_SCALE", "noise_w_scale"),
        ):
            value = os.getenv(env_name)
            if value:
                piper_args[argument_name] = float(value)
        return PiperTTS(**piper_args)
    if backend in {"kokoro-onnx", "kokoro_onnx"}:
        return KokoroONNXTTS(
            model_path=os.getenv(
                "KOKORO_ONNX_MODEL_PATH",
                "models/kokoro/kokoro-v1.0.onnx",
            ),
            voices_path=os.getenv(
                "KOKORO_ONNX_VOICES_PATH",
                "models/kokoro/voices-v1.0.bin",
            ),
            voice=os.getenv("KOKORO_ONNX_VOICE", "af_heart"),
            lang=os.getenv("KOKORO_ONNX_LANG", "en-us"),
            speed=float(os.getenv("KOKORO_ONNX_SPEED", "1.0")),
            sentence_pause=float(os.getenv("KOKORO_ONNX_SENTENCE_PAUSE", "0.25")),
            clause_pause=float(os.getenv("KOKORO_ONNX_CLAUSE_PAUSE", "0.1")),
        )
    raise ValueError(
        f"Unsupported VOICE_TTS_BACKEND={backend!r}; expected 'piper' or 'kokoro-onnx'"
    )

_DIRECT_INSTRUCTIONS = """You are Maya, a friendly and helpful voice assistant.

Speak naturally using simple language, contractions, and a mix of short and medium-length sentences.
Use commas and occasional sentence breaks to create natural pauses, but do not add filler words unless they genuinely help the conversation.
Vary sentence length and openings so replies do not sound like a repeated script.
Use a brief acknowledgement when it helps the listener, but do not begin every reply with "Certainly," "Sure," or "Absolutely."
Prefer spoken phrasing over written prose: avoid dense paragraphs, nested clauses, stage directions, and unnecessary formal transitions.
Begin every reply with a short, complete first sentence of roughly 4 to 10 words.
Do not use markdown, bullets, asterisks, headings, or special formatting.
Answer directly and avoid unnecessary explanations.
Use at most two short sentences for simple answers, and up to three short spoken sentences for a necessary topic list. For topic-list questions, name the retrieved topics naturally in a concise spoken list; do not replace them with only unit numbers.

Answer every question directly using your own knowledge and the conversation context.
Be warm, conversational, accurate, concrete, and concise.
Do not invent facts when uncertain.
Do not expand or define an acronym unless you know its meaning.
The user's words come from speech recognition and may contain errors. If a phrase looks wrong, infer the most likely intended meaning from context and the notes provided, rather than taking it literally."""

RAG_INSTRUCTIONS = _DIRECT_INSTRUCTIONS + """
Use the conversation context directly; do not mention retrieval internals."""

RAG_ENABLED_INSTRUCTIONS = _DIRECT_INSTRUCTIONS + """
When relevant notes are provided, use them for indexed-document and engineering-roadmap questions.
For greetings and general knowledge, answer directly without relying on notes."""


def _session_instructions() -> str:
    return RAG_ENABLED_INSTRUCTIONS if _rag_enabled() else RAG_INSTRUCTIONS


def _warm_local_llm_sync() -> None:
    """Prime llama.cpp's prompt/KV cache without adding a voice-path call."""
    if os.getenv("VOICE_LLM_WARMUP", "1").strip().lower() in {"0", "false", "no"}:
        return
    url = os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8080/v1").rstrip("/")
    payload = json.dumps(
        {
            "model": os.getenv("LOCAL_LLM_MODEL", "qwen2.5-3b-instruct-q8_0.gguf"),
            "messages": [
                {"role": "system", "content": _session_instructions()},
                {"role": "user", "content": "Ready?"},
            ],
            "stream": False,
            "max_tokens": 1,
            "temperature": 0,
        }
    ).encode()
    request = urllib.request.Request(
        f"{url}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started_at = time.perf_counter()
    try:
        with urllib.request.urlopen(
            request,
            timeout=float(os.getenv("VOICE_LLM_WARMUP_TIMEOUT_S", "3")),
        ) as response:
            response.read()
        logger.info(
            format_event(
                "PREWARM COMPLETE",
                component="local_llm",
                elapsed_s=time.perf_counter() - started_at,
            )
        )
    except Exception as exc:
        logger.warning(
            format_event(
                "LLM WARMUP SKIP",
                error=type(exc).__name__,
                elapsed_s=time.perf_counter() - started_at,
            )
        )


_SENTENCE_END = re.compile(r"[.!?](?:[\"'’”)]*)(?:\s+|$)")
_CLAUSE_END = re.compile(r"[,;:.!?](?:\s+|$)")
_MIN_SENTENCE_CHARS = int(os.getenv("VOICE_FIRST_SENTENCE_MIN_CHARS", "12"))
_MIN_SENTENCE_WORDS = int(os.getenv("VOICE_FIRST_SENTENCE_MIN_WORDS", "3"))
_FIRST_CHUNK_MAX_CHARS = int(os.getenv("VOICE_FIRST_CHUNK_MAX_CHARS", "96"))


def _cut_at_word(text: str, word_count: int) -> tuple[str | None, str]:
    """Split after ``word_count`` words without dropping the remaining text."""
    matches = list(re.finditer(r"\S+", text))
    if len(matches) < word_count:
        return None, text
    end = matches[word_count - 1].end()
    return text[:end].strip(), text[end:].lstrip()


def _cut_first_chunk(text: str) -> tuple[str | None, str]:
    """Keep the first TTS request short in both words and characters."""
    matches = list(re.finditer(r"\S+", text))
    if len(matches) < 4:
        return None, text
    for word_count in range(min(8, len(matches)), 3, -1):
        end = matches[word_count - 1].end()
        candidate = text[:end].strip()
        if len(candidate) <= _FIRST_CHUNK_MAX_CHARS:
            return candidate, text[end:].lstrip()
    return _cut_at_word(text, 4)


def _pop_chunk(text: str, *, first: bool = False, final: bool = False) -> tuple[str | None, str]:
    """Pop the earliest useful speech chunk from a streaming LLM buffer.

    The first chunk is deliberately short so Kokoro can begin synthesis quickly.
    Later chunks prefer sentence boundaries but are capped when a model emits a
    long unterminated sentence.
    """
    if final:
        chunk = text.strip()
        return (chunk or None), ""

    words = len(text.split())
    if first and words >= 4:
        for match in _CLAUSE_END.finditer(text):
            candidate = text[: match.end()].strip()
            if 4 <= len(candidate.split()) <= 8:
                if len(candidate) <= _FIRST_CHUNK_MAX_CHARS:
                    return candidate, text[match.end() :].lstrip()
                return _cut_first_chunk(text)

    for match in _SENTENCE_END.finditer(text):
        candidate = text[: match.end()].strip()
        if first and len(candidate.split()) >= 4:
            if len(candidate.split()) <= 8 and len(candidate) <= _FIRST_CHUNK_MAX_CHARS:
                return candidate, text[match.end() :].lstrip()
            return _cut_first_chunk(text)
        return candidate, text[match.end() :].lstrip()

    if first and words >= 4:
        return _cut_first_chunk(text)

    if not first and (len(text) >= 100 or words >= 18):
        clause_matches = list(_CLAUSE_END.finditer(text))
        if clause_matches:
            match = clause_matches[-1]
            candidate = text[: match.end()].strip()
            if len(candidate.split()) >= 4:
                return candidate, text[match.end() :].lstrip()
        if words >= 18:
            return _cut_at_word(text, 18)

    return None, text


def _pop_completed_sentence(text: str, *, final: bool = False) -> tuple[str | None, str]:
    """Return one safe sentence and the remaining text.

    Short fragments stay buffered while the model is streaming. At the end of
    generation, the remaining text is returned even when it has no punctuation.
    """
    if final:
        sentence = text.strip()
        return (sentence or None), ""

    for match in _SENTENCE_END.finditer(text):
        end = match.end()
        candidate = text[:end].strip()
        if (
            len(candidate) >= _MIN_SENTENCE_CHARS
            or len(candidate.split()) >= _MIN_SENTENCE_WORDS
        ):
            return candidate, text[end:].lstrip()
    return None, text


class TurnLatencyTable:
    """Collect supported LiveKit and RAG timings into one log row per turn.

    The values are the SDK's component timings plus the RAG retrieval timing.
    ``first_audio_e2e`` is an additive approximation (endpointing + retrieval
    + LLM TTFT + TTS TTFB), not a wall-clock trace.  The explicit flags make
    slow turns easy to find in a plain-text log stream while keeping the
    The emitted rows are human-readable and include both component timings and
    a wall-clock first-audio estimate.
    """

    _OUTLIER_THRESHOLDS = {
        "transcript": 0.8,
        "rag": 0.8,
        "llm_ttft": 1.0,
        "tts_ttfb": 2.0,
        "tts_rtf": 1.0,
        "first_audio_e2e": 3.0,
    }

    def __init__(self) -> None:
        self._turns: dict[str, dict[str, float | bool | int | str]] = defaultdict(dict)
        self._traces: dict[str, VoiceTrace] = {}
        self._logged_speech_ids: set[str] = set()
        self._pending_retrievals: list[tuple[object, float]] = []
        self.interruptions = 0
        self.cancellations = 0

    def record_retrieval(self, speech_id: str | None, retrieval) -> None:
        """Record retrieval now, attaching it to the speech metric when available."""
        completed_at = time.perf_counter()
        if speech_id is None:
            self._pending_retrievals.append((retrieval, completed_at))
            return
        self._attach_retrieval(str(speech_id), retrieval, completed_at)

    def _attach_retrieval(self, speech_id: str, retrieval, completed_at: float) -> None:
        set_log_context(turn=speech_id)
        trace = self._traces.setdefault(speech_id, VoiceTrace(speech_id))
        trace.mark_at("rag_started", completed_at - retrieval.total_s)
        trace.mark_at("rag_finished", completed_at)
        turn = self._turns[speech_id]
        turn.update(
            rag_embedding_s=retrieval.embedding_s,
            rag_database_s=retrieval.database_s,
            rag_total_s=retrieval.total_s,
        )
        logger.info(
            format_event(
                "RAG COMPLETE",
                turn=speech_id,
                embedding_s=retrieval.embedding_s,
                database_s=retrieval.database_s,
                total_s=retrieval.total_s,
                result_count=retrieval.result_count,
            )
        )

    def _attach_next_pending_retrieval(self, speech_id: str) -> None:
        if not self._pending_retrievals:
            return
        retrieval, completed_at = self._pending_retrievals.pop(0)
        self._attach_retrieval(speech_id, retrieval, completed_at)

    def record_metric(self, metric) -> None:
        if isinstance(metric, InterruptionMetrics):
            self.interruptions += metric.num_interruptions
            return

        speech_id = getattr(metric, "speech_id", None)
        if not speech_id:
            return
        speech_id = str(speech_id)
        self._attach_next_pending_retrieval(speech_id)
        set_log_context(turn=speech_id)
        trace = self._traces.setdefault(speech_id, VoiceTrace(speech_id))
        turn = self._turns[speech_id]
        if isinstance(metric, EOUMetrics):
            now = time.perf_counter()
            # Both SDK delays are measured from the same stopped-speaking anchor:
            # transcription_delay = final transcript - stop, while
            # end_of_utterance_delay = commit - stop. They must not be subtracted
            # from each other when reconstructing the wall-clock timeline.
            stopped_speaking_at = now - metric.end_of_utterance_delay
            trace.mark_at(
                "user_stopped_speaking",
                stopped_speaking_at,
            )
            trace.mark_at("stt_final", stopped_speaking_at + metric.transcription_delay)
            trace.mark("endpoint_complete")
            turn.update(
                transcript_s=metric.transcription_delay,
                endpoint_s=metric.end_of_utterance_delay,
            )
        elif isinstance(metric, LLMMetrics):
            now = time.perf_counter()
            trace.mark_at("llm_started", now - metric.ttft)
            trace.mark_at("llm_first_token", now - metric.duration + metric.ttft)
            trace.mark("llm_finished")
            turn["llm_ttft_s"] = float(turn.get("llm_ttft_s", 0.0)) + metric.ttft
            turn["llm_duration_s"] = float(turn.get("llm_duration_s", 0.0)) + metric.duration
            turn["cancelled"] = bool(turn.get("cancelled", False) or metric.cancelled)
        elif isinstance(metric, TTSMetrics):
            ttfb = max(0.0, metric.ttfb)
            now = time.perf_counter()
            tts_started_at = now - max(0.0, metric.duration)
            first_audio_at = tts_started_at + ttfb
            trace.mark_at("tts_started", tts_started_at)
            if not metric.cancelled:
                trace.mark_at("tts_first_audio", first_audio_at)
                trace.mark_at("first_audio_published", first_audio_at)
                logger.info(
                    format_event(
                        "FIRST AUDIO",
                        turn=speech_id,
                        tts_ttfb_s=ttfb,
                        wall_first_audio_s=trace.elapsed(
                            "user_stopped_speaking", "first_audio_published"
                        ),
                    )
                )
            turn.update(
                tts_ttfb_s=ttfb,
                tts_rtf=(metric.duration / metric.audio_duration if metric.audio_duration else 0.0),
                wall_first_audio_s=trace.elapsed(
                    "user_stopped_speaking", "first_audio_published"
                ),
                cancelled=bool(turn.get("cancelled", False) or metric.cancelled),
            )
            if metric.cancelled:
                self.cancellations += 1
            # A single speech can emit more than one TTS metric (for example,
            # when the response is split into multiple audio segments). Keep
            # one manager-facing row per speech instead of duplicate rows.
            if speech_id not in self._logged_speech_ids:
                self._logged_speech_ids.add(speech_id)
                self.log_turn(speech_id)

    def _outlier_flags(self, turn: dict[str, float | bool | int | str], e2e: float) -> list[str]:
        thresholds = self._OUTLIER_THRESHOLDS
        flags: list[str] = []
        if float(turn.get("transcript_s", 0.0)) > thresholds["transcript"]:
            flags.append("transcript")
        if float(turn.get("rag_total_s", 0.0)) > thresholds["rag"]:
            flags.append("rag")
        if float(turn.get("llm_ttft_s", 0.0)) > thresholds["llm_ttft"]:
            flags.append("llm_ttft")
        if float(turn.get("tts_ttfb_s", 0.0)) > thresholds["tts_ttfb"]:
            flags.append("tts_ttfb")
        if float(turn.get("tts_rtf", 0.0)) > thresholds["tts_rtf"]:
            flags.append("tts_rtf")
        if e2e > thresholds["first_audio_e2e"]:
            flags.append("first_audio_e2e")
        return flags

    def log_turn(self, speech_id: str) -> None:
        turn = self._turns[speech_id]
        trace = self._traces.get(speech_id)
        additive_e2e = sum(
            float(turn.get(key, 0.0))
            for key in ("endpoint_s", "rag_total_s", "llm_ttft_s", "tts_ttfb_s")
        )
        wall_e2e = turn.get("wall_first_audio_s")
        e2e = float(wall_e2e) if isinstance(wall_e2e, (int, float)) else additive_e2e
        turn["e2e_s"] = e2e
        flags = self._outlier_flags(turn, e2e)
        turn["outlier_flags"] = ",".join(flags)
        logger.info(
            format_event(
                "TURN COMPLETE",
                turn=speech_id,
                transcript_s=float(turn.get("transcript_s", 0.0)),
                endpoint_s=float(turn.get("endpoint_s", 0.0)),
                rag_s=float(turn.get("rag_total_s", 0.0)),
                llm_ttft_s=float(turn.get("llm_ttft_s", 0.0)),
                llm_duration_s=float(turn.get("llm_duration_s", 0.0)),
                tts_ttfb_s=float(turn.get("tts_ttfb_s", 0.0)),
                tts_rtf=float(turn.get("tts_rtf", 0.0)),
                wall_first_audio_s=turn.get("wall_first_audio_s"),
                e2e_s=e2e,
                additive_first_audio_s=additive_e2e,
                cancelled=bool(turn.get("cancelled", False)),
                interruptions=self.interruptions,
                cancellations=self.cancellations,
                outlier_flags=",".join(flags) or "none",
            ),
        )
        if trace is not None:
            logger.info(format_event("TURN TRACE", turn=speech_id, steps=">".join(trace.steps)))

    async def log_summary(self) -> None:
        """Emit a shutdown-safe aggregate row for the LiveKit callback."""
        e2e_values = sorted(
            float(turn["e2e_s"])
            for turn in self._turns.values()
            if isinstance(turn.get("e2e_s"), (int, float))
        )

        def percentile(values: list[float], p: float) -> float | None:
            if not values:
                return None
            index = min(len(values) - 1, round((len(values) - 1) * p))
            return values[index]

        logger.info(
            format_event(
                "SESSION SUMMARY",
                turns=len(self._logged_speech_ids),
                interruptions=self.interruptions,
                cancellations=self.cancellations,
                outlier_turns=sum(bool(turn.get("outlier_flags")) for turn in self._turns.values()),
                e2e_p50_s=percentile(e2e_values, 0.50),
                e2e_p95_s=percentile(e2e_values, 0.95),
            )
        )


def _is_rag_context_message(item) -> bool:
    return (
        getattr(item, "type", None) == "message"
        and getattr(item, "role", None) == "system"
        and getattr(item, "text_content", "").startswith("Relevant notes:")
    )


def _compact_chat_context(chat_ctx: llm.ChatContext) -> llm.ChatContext:
    """Retain stable instructions and only recent conversational items."""
    max_items = int(os.getenv("VOICE_HISTORY_MAX_ITEMS", "8"))
    rag_indices = [
        index for index, item in enumerate(chat_ctx.items) if _is_rag_context_message(item)
    ]
    last_user_index = max(
        (index for index, item in enumerate(chat_ctx.items) if getattr(item, "role", None) == "user"),
        default=-1,
    )
    # AgentActivity gives this generation a mutable copy. Keep only the note
    # appended after the current user message; older notes are never persisted.
    current_rag = []
    if rag_indices and rag_indices[-1] > last_user_index:
        current_rag = [chat_ctx.items[rag_indices[-1]]]
    instructions = [
        item
        for item in chat_ctx.items
        if item.type == "message" and item.role in ("system", "developer")
        and not _is_rag_context_message(item)
    ]
    conversation = [
        item
        for item in chat_ctx.items
        if item not in instructions and not _is_rag_context_message(item)
    ]
    if len(conversation) <= max_items + 4:
        return llm.ChatContext(items=[*instructions, *current_rag, *conversation])

    # Trim in chunks instead of dropping one item every turn. This keeps the
    # stable prompt prefix intact for several turns and preserves llama.cpp KV
    # prompt-cache reuse.
    selected: set[int] = set(range(max(0, len(conversation) - max_items), len(conversation)))
    call_indices = [i for i, item in enumerate(conversation) if item.type == "function_call"]
    if call_indices:
        call_index = call_indices[-1]
        call_item = conversation[call_index]
        selected.add(call_index)
        call_id = getattr(call_item, "call_id", None)
        for i, item in enumerate(conversation):
            if item.type == "function_call_output" and (
                call_id is None or getattr(item, "call_id", None) == call_id
            ):
                selected.add(i)
                break
        # Include only the first assistant response after the newest tool call.
        # Adding every later assistant message defeats the history cap.
        for i in range(call_index + 1, len(conversation)):
            if conversation[i].type == "message" and getattr(conversation[i], "role", None) == "assistant":
                selected.add(i)
                break
    kept = [item for i, item in enumerate(conversation) if i in selected]
    return llm.ChatContext(items=[*instructions, *current_rag, *kept])


def create_local_llm() -> openai.LLM:
    """Create the low-latency local llama-server LLM."""
    return openai.LLM(
        model=os.getenv("LOCAL_LLM_MODEL", "qwen2.5-3b-instruct-q8_0.gguf"),
        base_url=os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8080/v1"),
        api_key=os.getenv("LOCAL_LLM_API_KEY", "local"),
        temperature=0.0,
        parallel_tool_calls=False,
        max_completion_tokens=int(os.getenv("LOCAL_LLM_MAX_COMPLETION_TOKENS", "96")),
    )



class RagAgent(Agent):
    def __init__(self, instructions: str, latency: TurnLatencyTable):
        super().__init__(instructions=instructions)
        self._latency = latency

    async def _retrieve_for_turn(self, question: str):
        lowered = question.lower()
        broad_query = any(word in lowered for word in ("all", "list", "topics", "covered", "sections"))
        return await asyncio.to_thread(retrieve_with_timing, question, 4 if broad_query else 2)

    @staticmethod
    def _should_retrieve(question: str) -> bool:
        words = question.split()
        if len(words) < 3:
            return False
        lowered = question.strip().lower()
        greetings = (
            "hello",
            "hi ",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
            "how are you",
            "thanks",
            "thank you",
        )
        return not any(lowered == phrase or lowered.startswith(phrase) for phrase in greetings)

    @staticmethod
    def _compact_tool_context(context: str, *, limit: int | None = None) -> str:
        """Remove retrieval formatting before returning context to the LLM."""
        lines = [" ".join(line.split()) for line in context.splitlines()]
        compact = " ".join(line for line in lines if line)
        max_chars = limit or int(os.getenv("RAG_TOOL_MAX_CHARS", "700"))
        return compact[:max_chars]

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        """Retrieve relevant notes before generation, without a tool round trip."""
        if not _rag_enabled():
            return

        question = new_message.text_content.strip()
        if not self._should_retrieve(question):
            logger.info(format_event("RAG SKIP", reason="short_or_chitchat"))
            return

        logger.info(format_event("RAG START", question_chars=len(question)))
        try:
            retrieval = await asyncio.wait_for(self._retrieve_for_turn(question), timeout=0.4)
        except asyncio.TimeoutError:
            logger.warning(format_event("RAG SKIP", reason="timeout", timeout_s=0.4))
            return
        except Exception:
            logger.exception(format_event("RAG ERROR", error="retrieval_failed"))
            return

        self._latency.record_retrieval(None, retrieval)
        logger.info(
            format_event(
                "RAG RESULTS",
                chunks=retrieval.result_count,
                context_chars=len(retrieval.context),
                best_distance=retrieval.best_distance,
                relevant=retrieval.is_relevant,
            )
        )
        if not retrieval.is_relevant:
            logger.info(format_event("RAG SKIP", reason="irrelevant"))
            return

        compact_context = self._compact_tool_context(retrieval.context)
        if not compact_context:
            logger.info(format_event("RAG SKIP", reason="empty_context"))
            return

        turn_ctx.items = [item for item in turn_ctx.items if not _is_rag_context_message(item)]
        turn_ctx.add_message(role="system", content=f"Relevant notes: {compact_context}")
        logger.info(
            format_event(
                "RAG CONTEXT",
                chunks=retrieval.result_count,
                chars=len(compact_context),
            )
        )

    def llm_node(self, chat_ctx, tools, model_settings):
        compacted = _compact_chat_context(chat_ctx)
        rag_context_present = any(
            _is_rag_context_message(item) and bool(getattr(item, "text_content", "").strip())
            for item in compacted.items
        )
        logger.info(
            format_event(
                "LLM PROMPT",
                context_items=len(compacted.items),
                rag_context=rag_context_present,
            )
        )
        async def stream_with_sentence_flush():
            generation_id = uuid.uuid4().hex[:12]
            started = time.perf_counter()
            buffer = ""
            first_token_logged = False
            first_sentence_logged = False
            first_chunk = True

            logger.info(format_event("LLM START", generation=generation_id))
            try:
                async for chunk in super(RagAgent, self).llm_node(
                    compacted, [], model_settings
                ):
                    if isinstance(chunk, llm.ChatChunk):
                        delta = chunk.delta
                        if delta is None:
                            if chunk.usage is not None:
                                yield chunk
                            continue

                        content = delta.content or ""
                        if not content:
                            if chunk.usage is not None:
                                yield chunk
                            continue
                    else:
                        content = str(chunk)

                    if content and not first_token_logged:
                        first_token_logged = True
                        logger.info(
                            format_event(
                                "LLM FIRST TOKEN",
                                generation=generation_id,
                                elapsed_s=time.perf_counter() - started,
                            )
                        )

                    buffer += content
                    while True:
                        sentence, buffer = _pop_chunk(buffer, first=first_chunk)
                        if sentence is None:
                            break
                        first_chunk = False
                        if not first_sentence_logged:
                            first_sentence_logged = True
                            logger.info(
                                format_event(
                                    "LLM FIRST SENTENCE",
                                    generation=generation_id,
                                    elapsed_s=time.perf_counter() - started,
                                    chars=len(sentence),
                                )
                            )
                        logger.info(
                            format_event(
                                "LLM FLUSH",
                                generation=generation_id,
                                chars=len(sentence),
                            )
                        )
                        yield sentence
                        yield FlushSentinel()

                sentence, buffer = _pop_chunk(buffer, final=True)
                if sentence:
                    if not first_sentence_logged:
                        logger.info(
                            format_event(
                                "LLM FIRST SENTENCE",
                                generation=generation_id,
                                elapsed_s=time.perf_counter() - started,
                                chars=len(sentence),
                            )
                        )
                    logger.info(
                        format_event(
                            "LLM FLUSH",
                            generation=generation_id,
                            chars=len(sentence),
                            final=True,
                        )
                    )
                    yield sentence
            finally:
                logger.info(
                    format_event(
                        "LLM COMPLETE",
                        generation=generation_id,
                        elapsed_s=time.perf_counter() - started,
                    )
                )

        return stream_with_sentence_flush()


@server.rtc_session(agent_name="local-voice-agent")
async def entrypoint(ctx: agents.JobContext) -> None:
    await ctx.connect()
    job = getattr(getattr(ctx, "job", None), "id", None)
    room_name = getattr(ctx.room, "name", None)
    set_log_context(job=job or "unknown", room=room_name or "unknown")
    logger.info(format_event("SESSION CONNECTED"))
    resources = await _ensure_local_resources(ctx.proc)
    latency = TurnLatencyTable()
    my_agent = RagAgent(
        instructions=RAG_ENABLED_INSTRUCTIONS if _rag_enabled() else RAG_INSTRUCTIONS,
        latency=latency,
    )

    session_vad = ctx.proc.userdata.get("vad")
    if session_vad is None:
        session_vad = inference.VAD(
            min_speech_duration=0.05,
            min_silence_duration=0.3,
            activation_threshold=0.5,
        )
    session_turn_detector = ctx.proc.userdata.get("turn_detector")
    if session_turn_detector is None:
        session_turn_detector = inference.TurnDetector(version="v1-mini")
    local_stt = resources["stt"]
    ctx.add_shutdown_callback(local_stt.aclose)
    local_tts = resources["tts"]
    ctx.add_shutdown_callback(local_tts.aclose)
    local_llm = create_local_llm()
    session = AgentSession(
        vad=session_vad,
        turn_handling=TurnHandlingOptions(
            turn_detection=session_turn_detector,
            endpointing={
                "min_delay": float(os.getenv("VOICE_ENDPOINTING_MIN_DELAY", "0.25")),
                "max_delay": float(os.getenv("VOICE_ENDPOINTING_MAX_DELAY", "0.6")),
            },
            preemptive_generation={
                # RAG mutates the turn context in on_user_turn_completed().
                # Agents 1.8.11 invalidates any preemptive LLM call when that
                # happens, causing a wasted first generation and a second one
                # with the retrieved context. Keep it for the non-RAG path.
                "enabled": (
                    os.getenv("VOICE_PREEMPTIVE_GENERATION", "1").strip().lower()
                    in {"1", "true", "yes", "on"}
                    and not _rag_enabled()
                ),
                "max_retries": 1,
                "preemptive_tts": os.getenv("VOICE_PREEMPTIVE_TTS", "0").strip().lower()
                in {"1", "true", "yes", "on"},
            },
            interruption={
                "mode": "vad",
                "min_duration": 0.8,
                "min_words": 4,
                "resume_false_interruption": True,
            },
        ),
        stt=local_stt,
        tts=local_tts,
        llm=local_llm,

        conn_options=SessionConnectOptions(
            llm_conn_options=APIConnectOptions(
                max_retry=0,
                timeout=60.0,
                liveness_timeout=60.0,
            )
        ),
    )
    # session_usage_updated is aggregate-only in Agents 1.8.11; the per-turn
    # speech_id/TTFT/TTFB/EOU fields still arrive through this deprecated event.
    session.on("metrics_collected", lambda event: latency.record_metric(event.metrics))
    ctx.add_shutdown_callback(latency.log_summary)

    # The API dispatches this job before issuing the user's token. Do not mark
    # the room ready until the agent is connected and every session resource
    # has been constructed or reused from process prewarm. AgentSession.start
    # will then wait for the first remote participant before RoomIO begins the
    # greeting flow.
    await ctx.room.local_participant.set_attributes(
        {AGENT_READY_ATTRIBUTE: "true"}
    )
    logger.info(format_event("AGENT READY", room=room_name))
    await session.start(
        room=ctx.room,
        agent=my_agent,
    )

    await session.say(
        text="Hey, I am maya how may i help you today ?"
    )

if __name__ == "__main__":
    agents.cli.run_app(server)





