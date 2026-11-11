
from __future__ import annotations

import os
import logging
import resource
import sys
import threading
import time
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv
from voice_logging import format_event

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalResult:
    context: str
    embedding_s: float
    database_s: float
    result_count: int
    scores: tuple[float, ...] = ()

    @property
    def total_s(self) -> float:
        return self.embedding_s + self.database_s

    @property
    def best_distance(self) -> float | None:
        return min(self.scores) if self.scores else None

    @property
    def is_relevant(self) -> bool:
        if self.result_count <= 0 or not self.context.strip():
            return False
        if not self.scores:
            # Older/fake vector stores do not expose scores; preserve their
            # existing behavior while real PGVector results use the gate below.
            return True
        return self.best_distance <= float(os.getenv("RAG_MAX_DISTANCE", "0.65"))



_resources_lock = threading.RLock()
_embeddings = None
_vector_store = None


def _create_embeddings():
    import torch
    from langchain_huggingface import HuggingFaceEmbeddings

    torch.set_num_threads(int(os.getenv("EMBEDDING_CPU_THREADS", "2")))

    query_encode_kwargs = {
        "normalize_embeddings": True,
    }
    query_prompt = os.getenv("EMBEDDING_QUERY_PROMPT", "").strip()
    if query_prompt:
        query_encode_kwargs["prompt_name"] = query_prompt

    requested_device = os.getenv("EMBEDDING_DEVICE", "cpu").strip().lower()
    if requested_device != "cpu":
        logger.warning(format_event("EMBEDDING DEVICE OVERRIDE", requested=requested_device, using="cpu"))

    return HuggingFaceEmbeddings(
        model_name=os.getenv(
            "EMBEDDING_MODEL",
            "Snowflake/snowflake-arctic-embed-xs",
        ),
        model_kwargs={
            "device": "cpu",
            "local_files_only": True,
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
        query_encode_kwargs=query_encode_kwargs,
    )



def get_embeddings():
    """Return the process-wide embedding client, constructing it once."""
    global _embeddings
    with _resources_lock:
        if _embeddings is None:
            _embeddings = _create_embeddings()
        return _embeddings


def _create_vector_store(embeddings):
    from langchain_postgres import PGVector

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError("DATABASE_URL is not configured")

    return PGVector(
        embeddings=embeddings,
        collection_name=os.getenv(
            "RAG_COLLECTION",
            "local_knowledge",
        ),
        connection=database_url,
        use_jsonb=True,
    )


def get_vector_store():
    """Return the process-wide vector store, constructing it once."""
    global _vector_store
    with _resources_lock:
        if _vector_store is None:
            _vector_store = _create_vector_store(get_embeddings())
        return _vector_store



def index_documents(documents) -> None:
    if not documents:
        return

    get_vector_store().add_documents(documents)


def reset_collection() -> None:
    """Delete the configured collection before a full re-ingestion."""
    global _vector_store
    with _resources_lock:
        get_vector_store().delete_collection()
        _vector_store = None
        _cached_query_embedding.cache_clear()


@lru_cache(maxsize=128)
def _cached_query_embedding(question: str) -> tuple[float, ...]:
    """Avoid repeat encoder work for repeated voice questions within a worker."""
    return tuple(get_embeddings().embed_query(question))


def retrieve_with_timing(question: str, limit: int = 4) -> RetrievalResult:

    if not question.strip():
        return RetrievalResult("No question was provided.", 0.0, 0.0, 0)

    embedding_started_at = time.perf_counter()
    embedding = list(_cached_query_embedding(question))
    embedding_s = time.perf_counter() - embedding_started_at
    database_started_at = time.perf_counter()
    store = get_vector_store()
    scores: tuple[float, ...] = ()
    if hasattr(store, "similarity_search_with_score_by_vector"):
        scored_documents = store.similarity_search_with_score_by_vector(embedding, k=limit)
        documents = [document for document, _score in scored_documents]
        scores = tuple(float(score) for _document, score in scored_documents)
    else:
        documents = store.similarity_search_by_vector(embedding, k=limit)
    database_s = time.perf_counter() - database_started_at

    if not documents:
        return RetrievalResult(
            "No matching local documents were found.", embedding_s, database_s, 0, scores=scores
        )

    results = []

    for document in documents:
        source = document.metadata.get("source", "Unknown")
        page = document.metadata.get("page", "Unknown")

        results.append(
            f"""
Source: {source}
Page: {page}

Content:
{document.page_content}
"""
        )

    # Keep voice prompts small and predictable even if an indexed source contains
    # unusually large chunks. The retrieval ranking itself is unchanged.
    context = "\n\n".join(results)
    max_context_chars = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "6000"))
    return RetrievalResult(
        context=context[:max_context_chars],
        embedding_s=embedding_s,
        database_s=database_s,
        result_count=len(documents),
        scores=scores,
    )


def retrieve(question: str, limit: int = 4) -> str:
    """Compatibility wrapper for callers that need only retrieved context."""
    return retrieve_with_timing(question, limit=limit).context


def ensure_rag_ready() -> None:
    """Initialize and exercise all RAG dependencies before calls are accepted."""
    embeddings = get_embeddings()
    store = get_vector_store()
    embedding = list(embeddings.embed_query("warmup"))
    store.similarity_search_by_vector(embedding, k=1)
    rss_mb = _resident_memory_mb()
    warning_mb = float(os.getenv("RAG_RSS_WARNING_MB", "2048"))
    log = logger.warning if rss_mb > warning_mb else logger.info
    log(
        format_event(
            "RAG READY",
            rss_mb=rss_mb,
            rss_warning_mb=warning_mb,
        )
    )


def _resident_memory_mb() -> float:
    """Return peak process RSS in MB on macOS and Linux."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


def warmup_rag() -> None:
    """Backward-compatible name for the readiness gate."""
    ensure_rag_ready()


def reset_rag_resources_for_testing() -> None:
    """Clear process globals for isolated unit tests only."""
    global _embeddings, _vector_store
    with _resources_lock:
        _embeddings = None
        _vector_store = None
        _cached_query_embedding.cache_clear()
