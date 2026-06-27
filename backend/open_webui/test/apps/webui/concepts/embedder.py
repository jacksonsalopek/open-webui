"""Test embedder helpers for the concept graph acceptance harness.

The HybridRetriever requires both:
  1. A query embedding (set via RouterConfig.embed_fn before HybridRetriever
     is constructed in _route_explain_region).
  2. Concept embeddings on the store (set via Concept.embedding so
     GraphStore.vector_search can score candidates).

Production embedding lives in retrieval/utils.py and is async/awaitable, with
a model that may need a network round-trip. Acceptance tests cannot tolerate
either of those, so we provide a fast deterministic fake here and an optional
real sentence-transformers path gated by an env var.

Backfill uses ``GraphStore.list_concepts()`` and ``set_concept_embedding()``
— both Protocol primitives — rather than duck-typing store internals.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from collections.abc import Callable

log = logging.getLogger(__name__)

FAKE_EMBED_DIM = 64
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")


def _l2_normalize(vec: list[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return tuple(vec)
    return tuple(x / norm for x in vec)


def _token_vector(token: str, dim: int) -> list[float]:
    """Fixed ``dim``-float vector derived from a single token via blake2b."""
    bytes_needed = dim * 4
    digest = b""
    counter = 0
    while len(digest) < bytes_needed:
        digest += hashlib.blake2b(
            token.encode() + counter.to_bytes(2, "little"),
            digest_size=64,
        ).digest()
        counter += 1
    digest = digest[:bytes_needed]

    vec: list[float] = []
    for i in range(dim):
        chunk = digest[i * 4 : (i + 1) * 4]
        u = int.from_bytes(chunk, "big")
        vec.append((u / 0xFFFFFFFF) * 2.0 - 1.0)
    return vec


def fake_embedder(dim: int = FAKE_EMBED_DIM) -> Callable[[str], tuple[float, ...]]:
    """Return a deterministic, content-addressed embedder.

    The vector for ``text`` is the L2-normalized sum of per-token hash-derived
    vectors. Tokens are case-folded ASCII word fragments. Two texts that share
    tokens produce vectors that share direction; this is sufficient to make
    HybridRetriever return non-empty results and to give vector_search a
    meaningful ordering signal in tests.
    """

    def _embed(text: str) -> tuple[float, ...]:
        tokens = [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]
        if not tokens:
            return tuple(0.0 for _ in range(dim))

        summed = [0.0] * dim
        for token in tokens:
            for i, value in enumerate(_token_vector(token, dim)):
                summed[i] += value
        return _l2_normalize(summed)

    return _embed


def real_embedder() -> Callable[[str], tuple[float, ...]] | None:
    """Return a sentence-transformers embedder if ST_MODEL_NAME is set and loads.

    Returns None on any failure (import error, model load error, etc.); callers
    are expected to fall back to ``fake_embedder``.
    """
    model_name = os.environ.get("CONCEPT_GRAPH_ST_MODEL")
    if not model_name:
        return None
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        model = SentenceTransformer(model_name)
    except Exception as e:  # noqa: BLE001
        log.warning("real_embedder unavailable (%s): %s", model_name, e)
        return None

    def _embed(text: str) -> tuple[float, ...]:
        vec = model.encode(text, normalize_embeddings=True)
        return tuple(float(x) for x in vec.tolist())

    return _embed


def get_acceptance_embedder() -> tuple[Callable[[str], tuple[float, ...]], str]:
    """Return ``(embed_fn, label)`` for the acceptance harness.

    Prefers ``real_embedder()`` if available; otherwise ``fake_embedder()``.
    The label is ``"real:<model>"`` or ``"fake"``; tests should print it for
    diagnostics so a future maintainer can tell which mode the run used.
    """
    real = real_embedder()
    if real is not None:
        return real, f"real:{os.environ.get('CONCEPT_GRAPH_ST_MODEL')}"
    return fake_embedder(), "fake"


def embed_store_concepts(
    store,
    embed_fn: Callable[[str], tuple[float, ...]],
    *,
    overwrite: bool = False,
) -> int:
    """Populate ``concept.embedding`` for every concept in ``store``.

    Uses ``store.list_concepts()`` to iterate and ``set_concept_embedding()``
    to write each vector. Embeds ``concept.name`` (deliberately NOT ``snippet``
    or other fields — name is the canonical token for the concept and matches
    what queries embed).

    Skips concepts that already have a non-None embedding unless
    ``overwrite=True``.

    Returns the number of concepts whose embeddings were set/changed.
    """
    count = 0
    for concept in store.list_concepts():
        if concept.embedding is not None and not overwrite:
            continue
        new_embedding = embed_fn(concept.name)
        store.set_concept_embedding(concept.id, new_embedding)
        count += 1
    return count
