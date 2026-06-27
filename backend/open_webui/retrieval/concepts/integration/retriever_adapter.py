"""langchain BaseRetriever adapter for the concept-graph router.

Wraps router.retrieve(...) so it can plug into langchain's
EnsembleRetriever alongside BM25 + VectorSearchRetriever. Phase 2 W1
delivers the adapter; W2 wires it into query_doc_with_hybrid_search.

Output: langchain Document objects with metadata fields that match the
shape EnsembleRetriever's RRF expects, AND that downstream code in
retrieval/utils.py uses for chunk-hash dedup.

Documents emitted by this adapter carry a ``_chunk_hash`` metadata key
(sha256 hex of ``page_content``) so they survive the RRF dedup in
EnsembleRetriever (which uses ``id_key='_chunk_hash'``). The hash
contract matches retrieval.utils._content_hash exactly. After W4-B
enrichment, ``page_content`` for concept hits carries definition,
token, and provenance signal (see Enrichment policy below) so the
production cross-encoder reranker can score cg-docs meaningfully.

Enrichment policy (concept hits only; artifact hits keep ``artifact.path``):

1. Start with ``concept.name`` (always present).
2. If ``concept.definition`` is set (PHRASE-kind): append ``": " + definition``.
3. If ``original_tokens`` contains tokens distinct from ``concept.name``:
   append `` [tokens: tok1, tok2, …]`` (max 6 tokens, name-exact matches
   skipped, order preserved).
4. If provenance carries a source path (via ``_source_from_provenance``):
   append `` (in: <basename>)``.

The composed string is capped at 400 characters; overflow is truncated
with a trailing ``…`` so cross-encoder inputs stay bounded. ``metadata``
``concept_name`` remains the plain concept name (not enriched) for the
acceptance scorer.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from open_webui.retrieval.concepts.retrieve.base import RetrievalHit

log = logging.getLogger(__name__)

# Mirror retrieval.utils.CHUNK_HASH_KEY + _content_hash exactly so the
# Documents this adapter emits survive EnsembleRetriever's RRF dedup
# (which uses ``id_key='_chunk_hash'`` and expects sha256 hex digests).
# We re-implement locally rather than importing from retrieval.utils to
# avoid pulling that whole heavy module into the concept-graph subpath.
# Contract: must produce identical hash output to retrieval.utils._content_hash.
_CHUNK_HASH_KEY = '_chunk_hash'


def _content_hash(text: str) -> str:
    """SHA-256 hex digest of text — must match retrieval.utils._content_hash."""
    return hashlib.sha256(text.encode()).hexdigest()


def _source_from_provenance(provenance: Mapping[str, Any]) -> str | None:
    """Return the first artifact path hint from hit provenance, if any."""
    for key in ('artifact_path', 'named_in_path', 'source_path'):
        value = provenance.get(key)
        if isinstance(value, str) and value:
            return value

    for key in ('named_in_paths', 'artifact_paths'):
        value = provenance.get(key)
        if isinstance(value, (list, tuple)) and value:
            first = value[0]
            if isinstance(first, str) and first:
                return first

    return None


def _resolve_source(hit: RetrievalHit) -> str:
    if hit.artifact is not None:
        return hit.artifact.path

    if hit.concept is not None:
        from_provenance = _source_from_provenance(hit.provenance)
        if from_provenance is not None:
            return from_provenance
        return hit.concept.name

    return ''


_MAX_PAGE_CONTENT_LEN = 400


def _build_concept_page_content(hit: RetrievalHit) -> str:
    """Compose enriched page_content for a concept hit.

    W3 risk #4: cg-doc page_content was only the concept name (a single
    token), giving the W4 production cross-encoder reranker almost no
    signal vs. 1200-char BM25 chunks. This helper layers definition,
    distinct original_tokens, and provenance basename onto the name so
    ``RerankCompressor`` can score cg-docs meaningfully.
    """
    concept = hit.concept
    assert concept is not None

    content = concept.name
    if concept.definition is not None:
        content += ': ' + concept.definition

    distinct_tokens = [t for t in concept.original_tokens if t != concept.name]
    if distinct_tokens:
        capped = distinct_tokens[:6]
        content += f' [tokens: {", ".join(capped)}]'

    src = _source_from_provenance(hit.provenance)
    if src is not None:
        content += f' (in: {Path(src).name})'

    if len(content) > _MAX_PAGE_CONTENT_LEN:
        content = content[:_MAX_PAGE_CONTENT_LEN - 1] + '…'

    return content


def _hit_to_document(hit: RetrievalHit, *, collection_name: str | None) -> Document:
    if hit.concept is not None:
        concept_name = hit.concept.name
        concept_id = hit.concept.id
        page_content = _build_concept_page_content(hit)
    else:
        assert hit.artifact is not None
        concept_name = hit.artifact.path
        concept_id = hit.artifact.id
        page_content = hit.artifact.path

    metadata: dict[str, Any] = {
        'concept_name': concept_name,
        'concept_id': concept_id,
        'score': hit.score,
        'retriever': 'concept_graph',
        'source': _resolve_source(hit),
        _CHUNK_HASH_KEY: _content_hash(page_content),
    }
    if collection_name is not None:
        metadata['collection_name'] = collection_name

    return Document(page_content=page_content, metadata=metadata)


class ConceptGraphRetriever(BaseRetriever):
    """langchain BaseRetriever that delegates to concept-graph router.retrieve.

    Construction parameters (passed via __init__ kwargs since
    BaseRetriever uses pydantic):
      router_retrieve: Callable[[str, int], Sequence[RetrievalHit]]
        A bound retrieve function (so the adapter doesn't need to know
        about RouterConfig). Wave 2 wires this from a configured router.
      k: int
        Top-K to request from the router. Default 10.
      collection_name: str | None
        Optional pass-through for metadata (helps downstream identify
        which collection the hits came from).

    On invoke, the adapter:
      1. Calls router_retrieve(query, k).
      2. Converts each RetrievalHit into a langchain Document with:
         - page_content: enriched concept text via
           ``_build_concept_page_content`` (name + definition + tokens +
           provenance basename); artifact hits use ``artifact.path``.
         - metadata: dict including 'concept_name', 'concept_id',
           'score', 'retriever' (always 'concept_graph'), 'source'
           (first IS_NAMED_IN artifact path if available, else
           concept_name as a fallback).
      3. Returns the list.

    Errors are NOT propagated — log and return []. This adapter is a
    soft-fail signal in the ensemble.
    """

    router_retrieve: Callable[[str, int], Sequence[RetrievalHit]]
    k: int = 10
    collection_name: str | None = None

    model_config = {'arbitrary_types_allowed': True}

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        try:
            hits = self.router_retrieve(query, self.k)
            log.debug(
                'concept_graph_retriever query=%r k=%d hits=%d',
                query,
                self.k,
                len(hits),
            )
            return [
                _hit_to_document(hit, collection_name=self.collection_name)
                for hit in hits
            ]
        except Exception:
            log.warning(
                'concept_graph_retriever failed query=%r k=%d',
                query,
                self.k,
                exc_info=True,
            )
            return []
