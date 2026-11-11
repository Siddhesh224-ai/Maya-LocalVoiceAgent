"""File-only, human-readable logging for the local voice worker."""

from __future__ import annotations

import logging
import multiprocessing
import os
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


_log_context: ContextVar[dict[str, str]] = ContextVar("voice_log_context", default={})
_configured = False
_livekit_log_queue_patched = False


def format_event(event: str, **fields: Any) -> str:
    """Format a stable, readable event line without dumping raw objects."""
    parts = [event]
    for key, value in fields.items():
        if value is None:
            value = "none"
        elif isinstance(value, float):
            value = f"{value:.3f}"
        parts.append(f"{key}={value}")
    return " | ".join(parts)


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        context = _log_context.get()
        record.voice_job = context.get("job", "-")
        record.voice_room = context.get("room", "-")
        record.voice_turn = context.get("turn", "-")
        return True


class _SingleEmissionFilter(logging.Filter):
    """Suppress duplicate delivery when a CLI/root handler is re-added."""

    _record_marker = "_voice_log_emitted"

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, self._record_marker, False):
            return False
        setattr(record, self._record_marker, True)
        return True


class _ReadableFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        if record.exc_info:
            return message
        return message.replace("\x1b", "")


def set_log_context(**fields: object) -> None:
    """Set bounded job/session context for subsequent log records."""
    current = dict(_log_context.get())
    current.update({key: str(value) for key, value in fields.items() if value is not None})
    _log_context.set(current)


def configure_logging(level: int | str | None = None) -> Path:
    """Write all Python logging output to ``logfile.txt`` and nowhere else."""
    global _configured

    log_path = Path(os.getenv("VOICE_LOG_FILE", "logfile.txt"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_path,
        maxBytes=int(os.getenv("VOICE_LOG_MAX_BYTES", str(10 * 1024 * 1024))),
        backupCount=int(os.getenv("VOICE_LOG_BACKUP_COUNT", "3")),
        encoding="utf-8",
    )
    handler.setFormatter(
        _ReadableFormatter(
            "%(asctime)s | %(levelname)-8s | process=%(processName)s:%(process)d "
            "| job=%(voice_job)s | room=%(voice_room)s | turn=%(voice_turn)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handler.addFilter(_ContextFilter())
    handler.addFilter(_SingleEmissionFilter())

    root = logging.getLogger()
    # LiveKit configures handlers on both the root logger and named loggers.
    # Remove every inherited handler so a child worker cannot write duplicate
    # lines or leak colored logs back to the terminal.
    loggers = [root]
    loggers.extend(
        logger
        for logger in logging.Logger.manager.loggerDict.values()
        if isinstance(logger, logging.Logger)
    )
    removed_handlers: set[int] = set()
    for log in loggers:
        for existing in log.handlers[:]:
            log.removeHandler(existing)
            if id(existing) not in removed_handlers:
                existing.close()
                removed_handlers.add(id(existing))
        log.propagate = True
    root.addHandler(handler)
    root.setLevel(os.getenv("VOICE_LOG_LEVEL", "").upper() or level or "INFO")

    # Third-party debug logs are rarely useful for turn diagnosis and can drown
    # out the latency events. Errors still pass through to logfile.txt.
    for noisy_logger in ("httpx", "httpcore", "urllib3", "mlx", "transformers"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    logging.captureWarnings(True)
    _configured = True
    return log_path


def patch_livekit_cli_logging() -> None:
    """Prevent LiveKit's legacy CLI from adding a terminal log handler."""
    global _livekit_log_queue_patched
    try:
        from livekit.agents.cli import _legacy
        from livekit.agents.cli import cli as cli_module
        from livekit.agents.cli import log as cli_log
    except ImportError:  # pragma: no cover - only used in worker execution
        return

    def _configure_file_logger(_console, log_level) -> None:
        configure_logging(log_level)

    def _setup_file_logging(log_level, devmode=False, console=False, compact=False) -> None:
        del devmode, console, compact
        configure_logging(log_level)

    _legacy._configure_logger = _configure_file_logger
    cli_module.setup_logging = _setup_file_logging
    cli_log.setup_logging = _setup_file_logging

    # Agents 1.8.11 adds an IPC LogQueueHandler after the application module
    # configures logging in each job process. That queue forwards the same
    # record to the parent, where our file handler writes it a second time.
    # Job processes already write directly to the shared rotating file, so the
    # forwarding leg must be silent for file-only logging.
    if not _livekit_log_queue_patched:
        try:
            from livekit.agents.ipc.log_queue import LogQueueHandler
        except ImportError:  # pragma: no cover - only used by worker execution
            return
        original_emit = LogQueueHandler.emit

        def _emit_without_duplicate(self, record) -> None:
            if _configured and multiprocessing.current_process().name == "job_proc":
                return
            original_emit(self, record)

        LogQueueHandler.emit = _emit_without_duplicate
        _livekit_log_queue_patched = True


def is_configured() -> bool:
    return _configured
