"""Production wiring for the concept-graph extras kwargs of
``query_doc_with_hybrid_search`` and ``query_collection_with_hybrid_search``.

Both entry points accept a ``concept_graph_embed_fn`` (sync ``Callable[[str],
tuple[float, ...]]``) and a ``concept_graph_reranker`` (sync
``Callable[[str, Sequence[RetrievalHit]], list[RetrievalHit]]``). Production
handlers in ``routers/retrieval.py`` historically passed ``None`` for both
(W3.5 / W4-C scaffolds). This module supplies the real closures when
``app.state.ef`` is a sync SentenceTransformer-shape embedder, and degrades
gracefully to ``None`` for engines whose embedder is async/external.

The reranker uses ``make_name_only_cosine_scorer`` (W6.6-C: decouples reranker
semantics from CO_OCCURS_WITH-enriched ``concept.embedding`` values, preserving
the q09 reranked recovery). This supersedes the W4.5 deployment guidance
(``make_cosine_scorer``), which was written before W6.6-C landed.
"""
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from open_webui.retrieval.concepts.retrieve.base import RetrievalHit

log = logging.getLogger(__name__)


def build_sync_concept_graph_embed_fn(ef: Any) -> "Callable[[str], tuple[float, ...]] | None":
    """Return a sync embedder closure over ``ef`` or ``None`` if unavailable.

    Production ``app.state.ef`` is either:
      - a ``sentence_transformers.SentenceTransformer`` instance (engine ``''``),
      - ``None`` (no local model, engine is external like ``ollama``/``openai``).

    SentenceTransformer's ``.encode`` is CPU-bound and sync; we wrap it the
    same way the acceptance harness's ``real_embedder()`` does (normalize +
    tolist + tuple).

    Returns ``None`` if ``ef`` is None or lacks an ``.encode`` method. Callers
    must propagate ``None`` to ``concept_graph_embed_fn``; the cg hook treats
    ``None`` as "no sync embedder available; ``'ppr_blend_embed'`` / ``'catrag'``
    tiebreakers degrade to pure PPR".
    """
    if ef is None or not hasattr(ef, "encode"):
        return None

    def _embed_fn(text: str) -> "tuple[float, ...]":
        vec = ef.encode(text, normalize_embeddings=True)
        # SentenceTransformer.encode returns numpy.ndarray; .tolist() yields list[float].
        # Some backends may return a torch.Tensor — .tolist() handles both.
        return tuple(float(x) for x in vec.tolist())

    return _embed_fn


def build_concept_graph_reranker(
    embed_fn: "Callable[[str], tuple[float, ...]] | None",
) -> "Callable[[str, Sequence[RetrievalHit]], list[RetrievalHit]] | None":
    """Return the production cg-reranker closure or ``None`` if unavailable.

    Uses ``make_name_only_cosine_scorer`` (W6.6-C) — scores hits by re-embedding
    ``concept.name`` on the fly, decoupled from any rich embedding stored on
    the concept. Falls back to ``None`` when ``embed_fn`` is None (no sync
    embedder available — the cg path then uses PPR order without rerank).
    """
    if embed_fn is None:
        return None

    # Import inside to avoid pulling reranker module into routers/retrieval.py
    # import graph at module-load time.
    from open_webui.retrieval.concepts.retrieve.reranker import (
        make_name_only_cosine_scorer,
        rerank_hits,
    )

    scorer = make_name_only_cosine_scorer(query_embed_fn=embed_fn)

    def _reranker(query, hits):
        return rerank_hits(query, hits, scorer=scorer)

    return _reranker


def build_concept_graph_extras(app_state: Any) -> dict:
    """Return the kwargs dict for the ``concept_graph_*`` extras of the hybrid-
    search entry points, given ``app.state``.

    Reads ``app_state.ef`` (the raw embedder model) and returns a dict ready to
    be ``**``-splatted into the call site, supplying:
      - ``concept_graph_embed_fn``: sync embed closure (or ``None``).
      - ``concept_graph_reranker``: name-only-cosine rerank closure (or ``None``).

    Both are tied to the same sync embedder; if one is ``None``, both are.

    Does NOT supply ``concept_graph_tiebreaker``, ``concept_graph_embed_alpha``,
    ``concept_graph_catrag_alpha``, ``concept_graph_store`` — those are owned
    by other call-site wiring (W6.9 keeps tiebreaker=None for PPR default;
    store comes from ``app.state.concept_graph_store``).
    """
    ef = getattr(app_state, "ef", None)
    embed_fn = build_sync_concept_graph_embed_fn(ef)
    reranker = build_concept_graph_reranker(embed_fn)
    if embed_fn is None:
        log.debug(
            "build_concept_graph_extras: no sync embedder available "
            "(app.state.ef=%s); cg-embed-fn + cg-reranker disabled",
            type(ef).__name__ if ef is not None else "None",
        )
    else:
        log.debug(
            "build_concept_graph_extras: sync embedder wired "
            "(ef=%s); cg-reranker enabled (name-only cosine)",
            type(ef).__name__,
        )
    return {
        "concept_graph_embed_fn": embed_fn,
        "concept_graph_reranker": reranker,
    }
