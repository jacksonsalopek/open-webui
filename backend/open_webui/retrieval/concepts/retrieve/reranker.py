"""Pluggable reranker for concept-graph ``RetrievalHit`` lists.

Used as a per-retriever PRE-RRF re-ordering pass inside the wired retrieve
hook (``retrieval/utils.py``). Re-orders hits by a caller-supplied scorer so
the cg-retriever's top-K reflects semantic relevance, not PPR rank — this
addresses the W3.6-documented RRF interleaving drop of cg-ranks 11-20 in the
wired EnsembleRetriever path. The reranker is NOT a model — it's an
orchestrator. Concrete scorers (cosine over concept embeddings, cross-encoder
over text, etc.) are pluggable via the ``scorer`` callable.

Behaviors
---------
1. **Determinism.** Given the same inputs, must produce the same output. No
   floating-point comparisons that depend on summation order beyond what
   ``sorted()`` provides.
2. **Graceful degradation.** Scorer exceptions MUST NOT propagate; log a
   warning and return the input unchanged. Length mismatches from the scorer
   raise ``ValueError`` (programming error). Production retrieve path must
   not break on a model reload glitch.
3. **No store access from inside the reranker module.** Scorers may read
   ``hit.concept.embedding`` (already populated when the hit is constructed)
   but MUST NOT call ``store.list_concepts()`` or any store method. The
   reranker is on the hot path and a store lookup would blow the latency
   budget.
4. **Logging.** One ``log.debug`` line per ``rerank_hits`` call: query text
   length, hit count, top_n, top-3 score values (post-rerank). One
   ``log.warning`` on scorer failure with exception info.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence

from open_webui.retrieval.concepts.retrieve.base import RetrievalHit

log = logging.getLogger(__name__)

ScorerFn = Callable[[str, Sequence[RetrievalHit]], Sequence[float]]
"""A scorer takes (query_text, hits) and returns one score per hit, higher = more relevant."""


def rerank_hits(
    query: str,
    hits: Sequence[RetrievalHit],
    *,
    scorer: ScorerFn,
    top_n: int | None = None,
    stable: bool = True,
) -> list[RetrievalHit]:
    """Re-order ``hits`` by ``scorer(query, hits)`` descending.

    - If ``top_n`` is provided, truncate to top_n after reordering.
    - If ``stable=True``, ties preserve original input order (use sorted with
      negative scores as keys so Python's stable sort handles ties).
    - Returns a new list; does not mutate input.
    - Robust to empty input: returns [].
    - If scorer returns a different length than hits, raise ValueError with a
      clear message — this is a programming error, not a runtime fallback.
    - If scorer raises, log a warning with ``exc_info=True`` and return the input
      unchanged (preserving original order). The reranker MUST NOT crash the
      retrieve path on scorer failure.
    """
    if not hits:
        return []

    try:
        scores = list(scorer(query, hits))
    except Exception:
        log.warning(
            'rerank_hits scorer failed; returning input unchanged',
            exc_info=True,
        )
        return list(hits)

    if len(scores) != len(hits):
        raise ValueError(
            f'scorer returned {len(scores)} scores for {len(hits)} hits',
        )

    if stable:
        order = sorted(range(len(hits)), key=lambda i: -scores[i])
    else:
        order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)

    reranked = [hits[i] for i in order]

    if top_n is not None:
        reranked = reranked[:top_n]

    top_scores = [scores[i] for i in order[:3]]
    log.debug(
        'rerank_hits query_len=%d hit_count=%d top_n=%s top_scores=%s',
        len(query),
        len(hits),
        top_n,
        top_scores,
    )

    return reranked


def _cosine_similarity(
    query_vec: Sequence[float],
    hit_vec: Sequence[float],
) -> float:
    dot = sum(q * h for q, h in zip(query_vec, hit_vec))
    norm_q = math.sqrt(sum(q * q for q in query_vec))
    norm_h = math.sqrt(sum(h * h for h in hit_vec))
    if norm_q == 0.0 or norm_h == 0.0:
        return 0.0
    return dot / (norm_q * norm_h)


def make_cosine_scorer(
    query_embed_fn: Callable[[str], Sequence[float]],
    *,
    hit_embedding_fn: Callable[[RetrievalHit], Sequence[float] | None] | None = None,
) -> ScorerFn:
    """Return a ScorerFn that cosine-scores hits using their concept embeddings.

    - ``query_embed_fn`` is called once per ``rerank_hits`` invocation with the
      query text. Result is the query vector.
    - ``hit_embedding_fn`` defaults to:
      ``lambda h: h.concept.embedding if h.concept is not None else None``.
      Hits whose extracted embedding is None get score ``-inf`` (sink them to
      the bottom).
    - Cosine sim: ``dot(q, h) / (norm(q) * norm(h))``. If either norm is 0,
      score is 0.0 for that hit (treat zero vector as "no information").
    - Implementation must NOT import numpy or scipy — use Python math only.
      The cg subsystem keeps its numerical dependencies minimal.
    """
    if hit_embedding_fn is None:
        hit_embedding_fn = lambda h: h.concept.embedding if h.concept is not None else None  # noqa: E731

    def scorer(query: str, hits: Sequence[RetrievalHit]) -> list[float]:
        query_vec = query_embed_fn(query)
        result: list[float] = []
        for hit in hits:
            embedding = hit_embedding_fn(hit)
            if embedding is None:
                result.append(float('-inf'))
            else:
                result.append(_cosine_similarity(query_vec, embedding))
        return result

    return scorer


def make_text_scorer(
    cross_encoder_fn: Callable[[str, Sequence[str]], Sequence[float]],
    *,
    text_fn: Callable[[RetrievalHit], str] | None = None,
) -> ScorerFn:
    """Return a ScorerFn that text-scores hits via a cross-encoder.

    - ``cross_encoder_fn`` is the underlying scorer: takes a query string and a
      list of document texts, returns one score per text. Compatible with
      ``sentence_transformers.CrossEncoder.predict`` after a thin closure that
      converts the (query, texts) shape into the model's ``[(q, t), ...]`` shape.
    - ``text_fn`` defaults to: returns ``hit.concept.name`` if concept else
      ``hit.artifact.path`` if artifact else ``""``. Callers can provide a richer
      text extractor (e.g. one that concatenates concept name + first artifact
      snippet) — sibling W4-B is enriching the adapter's page_content but
      W4-A's default text_fn stays minimal/conservative.
    """
    if text_fn is None:
        def _default_text_fn(hit: RetrievalHit) -> str:
            if hit.concept is not None:
                return hit.concept.name
            if hit.artifact is not None:
                return hit.artifact.path
            return ''

        text_fn = _default_text_fn

    def scorer(query: str, hits: Sequence[RetrievalHit]) -> list[float]:
        texts = [text_fn(hit) for hit in hits]
        return list(cross_encoder_fn(query, texts))

    return scorer
