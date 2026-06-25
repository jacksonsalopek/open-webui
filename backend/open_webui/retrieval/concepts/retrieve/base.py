"""Generic retrieval primitive interface for the concept knowledge graph."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from open_webui.retrieval.concepts.schema import (
    Artifact,
    Concept,
    ConceptKind,
    EdgeType,
)
from open_webui.retrieval.concepts.store.protocol import GraphStore


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    """Substrate-agnostic query input. Subset of fields are populated
    depending on the dispatch lane (step 9 router decides).

    ``text`` is always required (the original NL query).
    ``embedding`` is the pre-computed query vector (None if the caller
    doesn't have one yet — some retrievers don't need it).
    ``seed_concept_ids`` is a hint from the router for graph-walk queries
    (e.g., a ``find_symbol`` query resolves a token → concept id and
    passes it as a seed).
    ``top_k`` bounds the result set.
    """

    text: str
    embedding: tuple[float, ...] | None = None
    seed_concept_ids: tuple[int, ...] = ()
    top_k: int = 10
    edge_types_filter: tuple[EdgeType, ...] | None = None
    kind_filter: tuple[ConceptKind, ...] | None = None


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    """One ranked result. ``concept`` and ``artifact`` are mutually
    exclusive — a hit is either a concept or an artifact (chunk).
    ``score`` is retriever-defined; comparable WITHIN one retriever's
    output but not necessarily across retrievers.
    ``provenance`` is a free-form dict for debugging (which retriever,
    which strategy, intermediate scores, etc.)."""

    concept: Concept | None
    artifact: Artifact | None
    score: float
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (self.concept is None) == (self.artifact is None):
            raise ValueError('Exactly one of concept/artifact must be set')


class Retriever(Protocol):
    """Generic retrieval primitive interface. Each concrete retriever
    implements ONE strategy (neighborhood walk, vector search, hybrid,
    etc.). Step 9 router dispatches a ``RetrievalQuery`` to the right
    retriever based on classified intent.

    Implementations MUST be deterministic given the same store state
    and query — for tests and for cache-invalidation reasoning.
    """

    name: str  # class attr; e.g. 'neighborhood', 'hybrid'

    def retrieve(self, query: RetrievalQuery, store: GraphStore) -> list[RetrievalHit]: ...
