"""langchain BaseRetriever adapter for the concept-graph router.

Wraps router.retrieve(...) so it can plug into langchain's
EnsembleRetriever alongside BM25 + VectorSearchRetriever. Phase 2 W1
delivers the adapter; W2 wires it into query_doc_with_hybrid_search.

Phase 2.5 W1 (A1) emits three bounded peer streams of langchain
Document objects instead of a single concept-name stream:

1. **Concept neighbors** (``max_concept_neighbors``, default 3) —
   enriched concept text via ``_build_concept_page_content``.
2. **File paths** (``max_file_paths``, default 3) — artifact paths
   resolved via ``store.list_artifacts_for_concept``.
3. **Code chunks** (``max_code_chunks``, default 8) — chunk text from
   ``chunk_lookup(artifact.path)``.

Each stream is independently capped and deduplicated. Documents carry a
``_chunk_hash`` metadata key (sha256 hex of ``page_content``) so they
survive the RRF dedup in EnsembleRetriever (which uses
``id_key='_chunk_hash'``). The hash contract matches
retrieval.utils._content_hash exactly.

De-token policy (concept-neighbor stream, CG-on path only — when
``store`` is present): skip neighbor docs whose ``page_content`` equals
the bare concept name (enrichment added nothing). PHRASE concepts and
atomics with distinct ``original_tokens`` or provenance survive. The
no-store backward-compat path keeps bare-atomic neighbors (they serve
as the proxy-gate signal).

Enrichment policy (concept-neighbor stream only; artifact hits in the
file-path stream keep ``artifact.path``):

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
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from open_webui.retrieval.concepts.retrieve.base import RetrievalHit
from open_webui.retrieval.concepts.schema import EdgeType

log = logging.getLogger(__name__)

# Mirror retrieval.utils.CHUNK_HASH_KEY + _content_hash exactly so the
# Documents this adapter emits survive EnsembleRetriever's RRF dedup
# (which uses ``id_key='_chunk_hash'`` and expects sha256 hex digests).
# We re-implement locally rather than importing from retrieval.utils to
# avoid pulling that whole heavy module into the concept-graph subpath.
# Contract: must produce identical hash output to retrieval.utils._content_hash.
_CHUNK_HASH_KEY = '_chunk_hash'
_ARTIFACT_LOOKAHEAD = 40
_TOKEN_SPLIT_RE = re.compile(r'[^a-z0-9]+', re.IGNORECASE)
# English function words (len >= 3) that inflate chunk scores without
# discriminating answer-bearing code (e.g. "the"/"before" in comments).
_QUERY_STOPWORDS = frozenset({
    'who', 'what', 'where', 'when', 'how', 'why', 'which',
    'the', 'and', 'for', 'are', 'was', 'were', 'has', 'have',
    'does', 'did', 'that', 'this', 'with', 'from', 'into', 'over',
    'under', 'about', 'before', 'after', 'then', 'than', 'also',
    'not', 'but', 'can', 'will', 'would', 'should', 'could',
})
_CALL_VERB_TOKENS = frozenset({'call', 'calls', 'calling', 'invoke', 'invokes', 'invoking'})
_INVOKE_ASYNC_RE = re.compile(r'(\w+)async\s*\(', re.IGNORECASE)
_CALL_SITE_RE = re.compile(r'await\s+[\w.]+\.\w+async\s*\(', re.IGNORECASE)


def _query_tokens(query: str) -> list[str]:
    """Distinct query tokens (lowercase, len >= 3), order preserved."""
    parts = _TOKEN_SPLIT_RE.split(query.lower())
    return list(
        dict.fromkeys(
            t for t in parts if len(t) >= 3 and t not in _QUERY_STOPWORDS
        )
    )


def _calls_token_matches(chunk_lower: str, q_tokens: list[str]) -> bool:
    """True when chunk invokes *Async whose stem matches a query noun (backup -> BackupAsync)."""
    for match in _INVOKE_ASYNC_RE.finditer(chunk_lower):
        stem = match.group(1).replace('_', '').lower()
        if not stem:
            continue
        for token in q_tokens:
            if len(token) < 4:
                continue
            if stem == token or stem.startswith(token) or token.startswith(stem):
                return True
    return False


def _token_in_chunk(
    token: str,
    chunk_lower: str,
    chunk_token_set: set[str],
    q_tokens: list[str],
) -> bool:
    if token in chunk_token_set or token in chunk_lower:
        return True
    if token in _CALL_VERB_TOKENS:
        return _calls_token_matches(chunk_lower, q_tokens)
    return False


def _invoke_noun_bonus(chunk_lower: str, q_tokens: list[str]) -> float:
    """Boost chunks whose *Async call stem matches a query noun (backup -> BackupAsync)."""
    bonus = 0.0
    q_set = set(q_tokens)
    for match in _INVOKE_ASYNC_RE.finditer(chunk_lower):
        stem = match.group(1).replace('_', '').lower()
        if not stem:
            continue
        for token in q_set:
            if len(token) < 4:
                continue
            if stem == token or stem.startswith(token) or token.startswith(stem):
                bonus += 0.2
                break
    return bonus


def _chunk_query_score(chunk_text: str, query: str) -> float:
    """Fraction of distinct query tokens (len >= 3) found in chunk text."""
    q_tokens = _query_tokens(query)
    if not q_tokens:
        return 0.0
    lowered = chunk_text.lower()
    chunk_token_set = set(_TOKEN_SPLIT_RE.split(lowered))
    covered = sum(
        1
        for token in q_tokens
        if _token_in_chunk(token, lowered, chunk_token_set, q_tokens)
    )
    base = covered / len(q_tokens)
    bonus = _invoke_noun_bonus(lowered, q_tokens)
    if _CALL_SITE_RE.search(lowered):
        bonus += 0.15
    return min(1.0, base + bonus)


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


def _concept_neighbor_documents(
    hits: Sequence[RetrievalHit],
    *,
    max_concept_neighbors: int,
    collection_name: str | None,
    drop_bare_atomic: bool = False,
) -> list[Document]:
    if max_concept_neighbors <= 0:
        return []
    seen_hashes: set[str] = set()
    result: list[Document] = []
    for hit in hits:
        if hit.concept is None:
            continue
        doc = _hit_to_document(hit, collection_name=collection_name)
        if drop_bare_atomic and doc.page_content == hit.concept.name:
            continue
        h = doc.metadata[_CHUNK_HASH_KEY]
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        doc.metadata['stream'] = 'neighbors'
        result.append(doc)
        if len(result) >= max_concept_neighbors:
            break
    return result


def _file_path_documents(
    hits: Sequence[RetrievalHit],
    *,
    store: Any,
    chunk_lookup: Callable[[str], Sequence[tuple[str, Mapping[str, Any]]]] | None,
    query: str,
    max_file_paths: int,
    collection_name: str | None,
) -> list[Document]:
    if store is None or max_file_paths <= 0:
        return []

    use_query_ranking = chunk_lookup is not None and bool(_query_tokens(query))

    if not use_query_ranking:
        seen_paths: set[str] = set()
        result: list[Document] = []

        for hit in hits:
            if len(result) >= max_file_paths:
                break

            if hit.concept is not None:
                try:
                    artifacts = store.list_artifacts_for_concept(
                        hit.concept.id,
                        edge_types=(EdgeType.IS_NAMED_IN,),
                        limit=max_file_paths,
                    )
                except (KeyError, Exception):
                    log.debug(
                        'list_artifacts_for_concept failed for concept %r',
                        hit.concept.name,
                        exc_info=True,
                    )
                    continue

                for artifact in artifacts:
                    path = artifact.path
                    if path in seen_paths:
                        continue
                    seen_paths.add(path)

                    metadata: dict[str, Any] = {
                        'concept_name': hit.concept.name,
                        'concept_id': hit.concept.id,
                        'concept_definition': hit.concept.definition,
                        'score': hit.score,
                        'retriever': 'concept_graph',
                        'source': path,
                        'stream': 'file_paths',
                        _CHUNK_HASH_KEY: _content_hash(path),
                    }
                    if collection_name is not None:
                        metadata['collection_name'] = collection_name

                    result.append(Document(page_content=path, metadata=metadata))
                    if len(result) >= max_file_paths:
                        break

            elif hit.artifact is not None:
                doc = _hit_to_document(hit, collection_name=collection_name)
                path = doc.page_content
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                doc.metadata['stream'] = 'file_paths'
                result.append(doc)
                if len(result) >= max_file_paths:
                    break

        return result

    # Query-relevance ranking: score each path by max chunk overlap.
    path_entries: dict[str, tuple[float, float, int, RetrievalHit]] = {}

    for hit in hits:
        paths_with_order: list[tuple[str, int]] = []

        if hit.concept is not None:
            try:
                artifacts = store.list_artifacts_for_concept(
                    hit.concept.id,
                    edge_types=(EdgeType.IS_NAMED_IN,),
                    limit=_ARTIFACT_LOOKAHEAD,
                )
            except (KeyError, Exception):
                log.debug(
                    'list_artifacts_for_concept failed for concept %r',
                    hit.concept.name,
                    exc_info=True,
                )
                continue

            paths_with_order = [(a.path, idx) for idx, a in enumerate(artifacts)]

        elif hit.artifact is not None:
            paths_with_order = [(hit.artifact.path, 0)]

        for path, art_idx in paths_with_order:
            try:
                chunks = chunk_lookup(path)
            except Exception:
                log.debug('chunk_lookup failed for %r', path, exc_info=True)
                continue

            path_score = 0.0
            for chunk_text, _ in chunks:
                path_score = max(path_score, _chunk_query_score(chunk_text, query))

            existing = path_entries.get(path)
            candidate = (path_score, hit.score, art_idx, hit)
            if existing is None or candidate[:3] > existing[:3]:
                path_entries[path] = candidate

    ranked = sorted(
        path_entries.items(),
        key=lambda item: (-item[1][0], -item[1][1], item[1][2]),
    )

    result: list[Document] = []
    for path, (path_score, _hit_score, _art_idx, hit) in ranked[:max_file_paths]:
        if hit.concept is not None:
            metadata = {
                'concept_name': hit.concept.name,
                'concept_id': hit.concept.id,
                'concept_definition': hit.concept.definition,
                'score': hit.score,
                'retriever': 'concept_graph',
                'source': path,
                'stream': 'file_paths',
                _CHUNK_HASH_KEY: _content_hash(path),
            }
        else:
            doc = _hit_to_document(hit, collection_name=collection_name)
            metadata = dict(doc.metadata)
            metadata['stream'] = 'file_paths'

        if collection_name is not None:
            metadata['collection_name'] = collection_name

        result.append(Document(page_content=path, metadata=metadata))

    return result


def _code_chunk_documents(
    hits: Sequence[RetrievalHit],
    *,
    store: Any,
    chunk_lookup: Callable[[str], Sequence[tuple[str, Mapping[str, Any]]]] | None,
    query: str,
    max_code_chunks: int,
    collection_name: str | None,
) -> list[Document]:
    if store is None or chunk_lookup is None or max_code_chunks <= 0:
        return []

    use_query_ranking = bool(_query_tokens(query))

    if not use_query_ranking:
        seen_hashes: set[str] = set()
        result: list[Document] = []

        for hit in hits:
            if hit.concept is None:
                continue

            try:
                artifacts = store.list_artifacts_for_concept(
                    hit.concept.id,
                    edge_types=(EdgeType.IS_NAMED_IN,),
                    limit=None,
                )
            except (KeyError, Exception):
                log.debug(
                    'list_artifacts_for_concept failed for concept %r',
                    hit.concept.name,
                    exc_info=True,
                )
                continue

            for artifact in artifacts:
                try:
                    chunks = chunk_lookup(artifact.path)
                except Exception:
                    log.debug(
                        'chunk_lookup failed for %r',
                        artifact.path,
                        exc_info=True,
                    )
                    continue

                for chunk_text, chunk_metadata in chunks:
                    h = _content_hash(chunk_text)
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)

                    source = (
                        chunk_metadata.get('source', artifact.path)
                        if isinstance(chunk_metadata, Mapping)
                        else artifact.path
                    )
                    metadata: dict[str, Any] = {
                        **dict(chunk_metadata),
                        'concept_name': hit.concept.name,
                        'concept_id': hit.concept.id,
                        'concept_definition': hit.concept.definition,
                        'score': hit.score,
                        'retriever': 'concept_graph',
                        'source': source,
                        'stream': 'code_chunks',
                        _CHUNK_HASH_KEY: h,
                    }
                    if collection_name is not None:
                        metadata['collection_name'] = collection_name

                    result.append(Document(page_content=chunk_text, metadata=metadata))
                    if len(result) >= max_code_chunks:
                        return result

        return result

    candidates: list[
        tuple[float, float, int, str, Mapping[str, Any], RetrievalHit, str]
    ] = []

    for hit in hits:
        if hit.concept is None:
            continue

        try:
            artifacts = store.list_artifacts_for_concept(
                hit.concept.id,
                edge_types=(EdgeType.IS_NAMED_IN,),
                limit=_ARTIFACT_LOOKAHEAD,
            )
        except (KeyError, Exception):
            log.debug(
                'list_artifacts_for_concept failed for concept %r',
                hit.concept.name,
                exc_info=True,
            )
            continue

        for art_idx, artifact in enumerate(artifacts):
            try:
                chunks = chunk_lookup(artifact.path)
            except Exception:
                log.debug(
                    'chunk_lookup failed for %r',
                    artifact.path,
                    exc_info=True,
                )
                continue

            for chunk_text, chunk_metadata in chunks:
                chunk_meta = chunk_metadata if isinstance(chunk_metadata, Mapping) else {}
                q_score = _chunk_query_score(chunk_text, query)
                candidates.append(
                    (q_score, hit.score, art_idx, chunk_text, chunk_meta, hit, artifact.path),
                )

    if candidates:
        max_q = max(c[0] for c in candidates)
        if max_q > 0:
            candidates = [c for c in candidates if c[0] > 0]

    candidates.sort(
        key=lambda c: (
            -c[0],
            0 if _CALL_SITE_RE.search(c[3].lower()) else 1,
            -c[1],
            c[2],
        ),
    )

    seen_hashes: set[str] = set()
    result: list[Document] = []
    for q_score, hit_score, _art_idx, chunk_text, chunk_metadata, hit, artifact_path in candidates:
        h = _content_hash(chunk_text)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        source = chunk_metadata.get('source', artifact_path) if chunk_metadata else artifact_path
        metadata: dict[str, Any] = {
            **dict(chunk_metadata),
            'concept_name': hit.concept.name,
            'concept_id': hit.concept.id,
            'concept_definition': hit.concept.definition,
            'score': hit.score,
            'retriever': 'concept_graph',
            'source': source,
            'stream': 'code_chunks',
            _CHUNK_HASH_KEY: h,
        }
        if collection_name is not None:
            metadata['collection_name'] = collection_name

        result.append(Document(page_content=chunk_text, metadata=metadata))
        if len(result) >= max_code_chunks:
            break

    return result


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
      store: GraphStore | None
        Graph store for ``list_artifacts_for_concept`` (file-path and
        code-chunk streams). When None, only the concept-neighbor
        stream is emitted.
      chunk_lookup: Callable[[str], Sequence[tuple[str, Mapping]]] | None
        Resolves artifact paths to ``(chunk_text, chunk_metadata)``
        pairs from ``collection_result``. Required for the code-chunk
        stream.
      max_concept_neighbors: int
        Cap on concept-neighbor stream Documents (default 3).
      max_file_paths: int
        Cap on file-path stream Documents (default 3).
      max_code_chunks: int
        Cap on code-chunk stream Documents (default 8).

    On invoke, the adapter:
      1. Calls router_retrieve(query, k).
      2. Splits hits into concept vs artifact.
      3. Emits three bounded peer streams (neighbors + file_paths +
         code_chunks) concatenated in that order.

    Errors are NOT propagated — log and return []. This adapter is a
    soft-fail signal in the ensemble.
    """

    router_retrieve: Callable[[str, int], Sequence[RetrievalHit]]
    k: int = 10
    collection_name: str | None = None
    # Phase 2.5 W1 (A1) three-stream output contract:
    store: Any = None  # GraphStore | None
    chunk_lookup: Callable[[str], Sequence[tuple[str, Mapping[str, Any]]]] | None = None
    max_concept_neighbors: int = 3
    max_file_paths: int = 3
    max_code_chunks: int = 8

    model_config = {'arbitrary_types_allowed': True}

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        try:
            hits = self.router_retrieve(query, self.k)
        except Exception:
            log.warning(
                'concept_graph_retriever failed query=%r k=%d',
                query,
                self.k,
                exc_info=True,
            )
            return []

        concept_hits = [h for h in hits if h.concept is not None]
        artifact_hits = [h for h in hits if h.artifact is not None]

        neighbors = _concept_neighbor_documents(
            concept_hits,
            max_concept_neighbors=self.max_concept_neighbors,
            collection_name=self.collection_name,
            drop_bare_atomic=self.store is not None,
        )
        file_paths = _file_path_documents(
            concept_hits + artifact_hits,
            store=self.store,
            chunk_lookup=self.chunk_lookup,
            query=query,
            max_file_paths=self.max_file_paths,
            collection_name=self.collection_name,
        )
        code_chunks = _code_chunk_documents(
            concept_hits,
            store=self.store,
            chunk_lookup=self.chunk_lookup,
            query=query,
            max_code_chunks=self.max_code_chunks,
            collection_name=self.collection_name,
        )
        return neighbors + file_paths + code_chunks
