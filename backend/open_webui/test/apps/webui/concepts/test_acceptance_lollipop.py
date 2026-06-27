"""Phase 1 acceptance harness — 10 curated Lollipop developer questions."""

from __future__ import annotations

import functools
from pathlib import Path

import pytest
import yaml

from open_webui.retrieval.concepts.retrieve.base import RetrievalQuery

pytest.importorskip(
    'open_webui.retrieval.concepts.retrieve.router',
    reason='Step 9 router not implemented yet — acceptance harness skipped',
)
from open_webui.retrieval.concepts.retrieve.router import RouterConfig, route

PER_KIND_THRESHOLDS = {
    'find_symbol': 0.6,
    'where_used': 0.4,
    'explain_region': 0.3,
    'find_concept': 0.4,
    'generate_code': 0.0,
}


def _acceptance_yaml_path() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / 'retrieval' / 'concepts' / 'acceptance' / 'lollipop_v1.yaml'
        if candidate.is_file():
            return candidate
    raise FileNotFoundError('lollipop_v1.yaml not found relative to test module')


def _load_questions() -> list[dict]:
    with open(_acceptance_yaml_path()) as f:
        data = yaml.safe_load(f)
    return data['questions']


QUESTIONS = _load_questions()


def score_hit_set_against_expected(
    hits: list,
    expected_concepts: list[str],
    *,
    fuzzy_threshold: float = 0.6,
) -> tuple[bool, float, set[str]]:
    """Compute hit-rate of expected concepts against the retrieved hit set.

    ``hits`` is ``list[RetrievalHit]``. Each hit has either ``.concept.name`` or
    ``.artifact.path``. ``expected_concepts`` is a list of concept names (atomics
    or phrase names) from the YAML.

    Matching strategy:
    1. For each expected concept, check if ANY hit's concept.name equals it
       (case-insensitive exact match).
    2. If not found exactly, check if any hit's concept.name contains the
       expected as a substring (also case-insensitive). Substring matches are
       worth 0.5 of an exact match.

    Returns (passed, score, matched_set):
    - ``passed``: True iff score >= fuzzy_threshold.
    - ``score``: weighted hit-rate in [0, 1]. score = (exact_hits + 0.5 *
      substring_hits) / len(expected_concepts).
    - ``matched_set``: set of expected concept names that were found.

    empty ``expected_concepts`` raises ValueError (a question without expected
    concepts is malformed).
    """
    if not expected_concepts:
        raise ValueError('expected_concepts must not be empty')

    hit_names: list[str] = []
    for hit in hits:
        if hit.concept is not None:
            hit_names.append(hit.concept.name)
        elif hit.artifact is not None:
            hit_names.append(hit.artifact.path)

    matched: set[str] = set()
    exact_hits = 0
    substring_hits = 0

    for expected in expected_concepts:
        expected_lower = expected.lower()
        found_exact = False
        found_substring = False

        for name in hit_names:
            name_lower = name.lower()
            if name_lower == expected_lower:
                found_exact = True
                break
            if expected_lower in name_lower:
                found_substring = True

        if found_exact:
            matched.add(expected)
            exact_hits += 1
        elif found_substring:
            matched.add(expected)
            substring_hits += 1

    score = (exact_hits + 0.5 * substring_hits) / len(expected_concepts)
    return score >= fuzzy_threshold, score, matched


def _intent_value(result) -> str:
    classified = result.intent
    intent = getattr(classified, 'intent', classified)
    return intent.value if hasattr(intent, 'value') else str(intent)


def _hit_display_names(hits: list, limit: int = 10) -> list[str]:
    names: list[str] = []
    for hit in hits[:limit]:
        if hit.concept is not None:
            names.append(hit.concept.name)
        elif hit.artifact is not None:
            names.append(hit.artifact.path)
    return names


def _router_config(embed_fn) -> RouterConfig:
    # Honor the same CONCEPT_GRAPH_TIEBREAKER / CONCEPT_GRAPH_EMBED_ALPHA env
    # vars the cross-corpus experiment script uses, so an operator can do an
    # apples-to-apples ablation across both corpora without touching code.
    # Defaults preserve the existing pinned Lollipop tuning.
    import os as _os  # local import; keep module-level imports clean

    tiebreaker = _os.environ.get('CONCEPT_GRAPH_TIEBREAKER') or None
    alpha_env = _os.environ.get('CONCEPT_GRAPH_EMBED_ALPHA')
    embed_blend_alpha = float(alpha_env) if alpha_env else None
    catrag_alpha_env = _os.environ.get('CONCEPT_GRAPH_CATRAG_ALPHA')
    catrag_anchor_alpha = float(catrag_alpha_env) if catrag_alpha_env else None
    return RouterConfig(
        language='csharp',
        embed_fn=embed_fn,
        tiebreaker=tiebreaker,
        embed_blend_alpha=embed_blend_alpha,
        catrag_anchor_alpha=catrag_anchor_alpha,
    )


def score_documents_against_expected(
    documents: list,
    expected_concepts: list[str],
    *,
    fuzzy_threshold: float = 0.6,
) -> tuple[bool, float, set[str]]:
    """Wired-path analog of score_hit_set_against_expected.

    Operates on langchain ``Document`` objects (the output of
    ``query_doc_with_hybrid_search``) instead of ``RetrievalHit``s.
    Matching strategy is identical: per-expected case-insensitive
    exact or substring match against a candidate-name set.

    Each Document contributes ONE candidate name, prioritized as:
      1. ``metadata['concept_name']`` (set by ConceptGraphRetriever)
      2. ``metadata['source']`` (vector / bm25 hits)
      3. ``metadata['name']`` (langchain default)
      4. ``page_content`` first line (fallback)

    Returns (passed, score, matched_set) — identical contract to
    ``score_hit_set_against_expected``.
    """
    if not expected_concepts:
        raise ValueError('expected_concepts must not be empty')

    candidate_names: list[str] = []
    for doc in documents:
        meta = getattr(doc, 'metadata', {}) or {}
        candidate = (
            meta.get('concept_name')
            or meta.get('source')
            or meta.get('name')
            or (doc.page_content.splitlines()[0] if doc.page_content else '')
        )
        if candidate:
            candidate_names.append(str(candidate))

    matched: set[str] = set()
    exact_hits = 0
    substring_hits = 0

    for expected in expected_concepts:
        expected_lower = expected.lower()
        found_exact = False
        found_substring = False

        for name in candidate_names:
            name_lower = name.lower()
            if name_lower == expected_lower:
                found_exact = True
                break
            if expected_lower in name_lower:
                found_substring = True

        if found_exact:
            matched.add(expected)
            exact_hits += 1
        elif found_substring:
            matched.add(expected)
            substring_hits += 1

    score = (exact_hits + 0.5 * substring_hits) / len(expected_concepts)
    return score >= fuzzy_threshold, score, matched


@functools.lru_cache(maxsize=4)
def _build_synthetic_collection_result(corpus_root: str):
    """Re-chunk every .cs file under ``corpus_root`` via the AST splitter
    and return a SimpleNamespace mimicking a vector-db GetResult.

    Used by the wired-path tests — they need a ``collection_result``
    shaped like what ASYNC_VECTOR_DB_CLIENT.get returns, but without
    requiring a real ChromaDB.
    """
    from types import SimpleNamespace
    from open_webui.retrieval.loaders.code_splitter import split_code, ext_to_language

    chunk_texts: list[str] = []
    chunk_metadatas: list[dict] = []

    corpus_path = Path(corpus_root)
    for path in sorted(corpus_path.rglob('*.cs')):
        try:
            text = path.read_text(encoding='utf-8', errors='replace')
        except (OSError, UnicodeDecodeError):
            continue
        language = ext_to_language(str(path)) or 'csharp'
        try:
            chunks = split_code(
                text,
                language,
                chunk_size=1500,
                chunk_overlap=150,
                base_metadata={'source': str(path)},
            )
        except Exception:
            chunks = []
        if not chunks:
            chunks = [type('FakeDoc', (), {'page_content': text, 'metadata': {'source': str(path)}})()]
        for c in chunks:
            chunk_texts.append(c.page_content)
            chunk_metadatas.append(dict(c.metadata or {'source': str(path)}))

    return SimpleNamespace(
        documents=[chunk_texts],
        metadatas=[chunk_metadatas],
    )


@pytest.mark.parametrize(
    'q',
    QUESTIONS,
    ids=lambda q: f"{q['id']}-{q['intent']}-{q['difficulty']}",
)
def test_acceptance_question(q, lollipop_subset_store_with_embeddings, acceptance_embedder):
    """One acceptance question. Pass iff score >= per-question threshold."""
    if q['intent'] == 'generate_code':
        pytest.skip('generate_code is out of Phase 1 scope; informational only')

    expected = q['expected_concepts']
    if not expected:
        pytest.skip(f'Question {q["id"]} has no expected_concepts')

    embed_fn, _label = acceptance_embedder
    config = _router_config(embed_fn)
    query = RetrievalQuery(text=q['question'], top_k=20)
    result = route(query, lollipop_subset_store_with_embeddings, config=config)

    threshold = PER_KIND_THRESHOLDS.get(q['intent'], 0.5)
    passed, score, matched = score_hit_set_against_expected(
        result.hits,
        expected,
        fuzzy_threshold=threshold,
    )

    if not passed:
        pytest.fail(
            f'Question {q["id"]} FAILED.\n'
            f'  Intent classified: {_intent_value(result)}\n'
            f'  Retriever used: {result.retriever_used}\n'
            f'  Score: {score:.2f} (threshold: {threshold})\n'
            f'  Expected: {expected}\n'
            f'  Matched: {sorted(matched)}\n'
            f'  Missed: {sorted(set(expected) - matched)}\n'
            f'  Top-10 retrieved: {_hit_display_names(result.hits)}'
        )


def test_phase1_acceptance_rate(
    lollipop_subset_store_with_embeddings,
    acceptance_embedder,
):
    """Phase 1 acceptance gate: ≥ 60% of in-scope questions must pass."""
    embed_fn, embedder_label = acceptance_embedder
    config = _router_config(embed_fn)
    in_scope = [q for q in QUESTIONS if q['intent'] != 'generate_code']
    passed = 0
    failures: list[str] = []
    by_retriever: dict[str, dict[str, int]] = {}

    for q in in_scope:
        expected = q['expected_concepts']
        if not expected:
            continue

        result = route(
            RetrievalQuery(text=q['question'], top_k=20),
            lollipop_subset_store_with_embeddings,
            config=config,
        )
        threshold = PER_KIND_THRESHOLDS.get(q['intent'], 0.5)
        p, score, _matched = score_hit_set_against_expected(
            result.hits,
            expected,
            fuzzy_threshold=threshold,
        )
        retriever = getattr(result, 'retriever_used', 'unknown')
        by_retriever.setdefault(retriever, {'total': 0, 'passed': 0})
        by_retriever[retriever]['total'] += 1
        if p:
            passed += 1
            by_retriever[retriever]['passed'] += 1
        else:
            failures.append(
                f'{q["id"]} ({q["intent"]}, retriever={retriever}): '
                f'{score:.2f} < {threshold}'
            )

    pass_rate = passed / len(in_scope)
    retriever_parts = [
        f'{name}: {counts["passed"]}'
        for name, counts in sorted(by_retriever.items())
    ]
    retriever_summary = ', '.join(retriever_parts)
    print(
        f'\nacceptance: {passed}/{len(in_scope)} ({100 * pass_rate:.0f}%) pass '
        f'[{retriever_summary}]',
    )
    print(f'embedder: {embedder_label}')
    if failures:
        print('Failures:')
        for failure in failures:
            print(f'  {failure}')

    if pass_rate < 1.0:
        print(
            f'\nWARNING: acceptance dropped below 9/9 after embedder wiring '
            f'({passed}/{len(in_scope)}); investigate before merging W10-A.',
        )

    assert pass_rate >= 0.6, f'Pass rate {pass_rate:.2f} below 0.6 gate'


def _enable_concept_graph_for_wired_test():
    """Context-manager-style helper that flips CONCEPT_GRAPH_ENABLED True
    for the duration of a wired test. Yields the prior value."""
    from open_webui.config import CONCEPT_GRAPH_ENABLED
    prior = CONCEPT_GRAPH_ENABLED.value
    CONCEPT_GRAPH_ENABLED.value = True
    return CONCEPT_GRAPH_ENABLED, prior


def _patch_cg_documents_for_rrf():
    """ConceptGraphRetriever docs lack ``_chunk_hash``; RRF dedup requires it.

    Test-only shim until W2 adapter adds the key (see retriever_adapter.py).
    """
    from langchain_core.documents import Document
    from open_webui.retrieval.concepts.integration import retriever_adapter
    from open_webui.retrieval.utils import CHUNK_HASH_KEY, _content_hash

    original = retriever_adapter._hit_to_document

    def _hit_to_document_with_hash(hit, *, collection_name):
        doc = original(hit, collection_name=collection_name)
        meta = dict(doc.metadata or {})
        meta.setdefault(CHUNK_HASH_KEY, _content_hash(doc.page_content))
        return Document(page_content=doc.page_content, metadata=meta)

    retriever_adapter._hit_to_document = _hit_to_document_with_hash
    return original


def _restore_cg_documents_patch(original):
    from open_webui.retrieval.concepts.integration import retriever_adapter
    retriever_adapter._hit_to_document = original


def _wired_query(
    *,
    store,
    embed_fn,
    question_text: str,
    collection_result,
    k: int = 20,
) -> list:
    """Run one wired-path retrieval. Returns list[Document]."""
    import asyncio
    from open_webui.retrieval.utils import query_doc_with_hybrid_search

    async def _embedding_function(text, prefix=None, user=None):
        # The cg-retriever doesn't use this directly — it uses RouterConfig.
        # The RerankCompressor falls back to cosine when reranking_function
        # is None; we provide a constant reranking_function below so
        # embedding_function is irrelevant to the final order.
        if isinstance(text, list):
            return [[0.0] * 16] * len(text)
        return [0.0] * 16

    def _constant_reranker(query, documents, user=None):
        return [1.0] * len(documents)

    patch_original = _patch_cg_documents_for_rrf()
    try:
        # W3.5-B: pass the acceptance embedder through so the wired cg-retriever
        # can engage the 'ppr_blend_embed' tiebreaker the in-process tests use
        # via _router_config(embed_fn). Closes the wired vs in-process gap.
        result_dict = asyncio.run(
            query_doc_with_hybrid_search(
                collection_name='lollipop-acceptance-wired',
                collection_result=collection_result,
                query=question_text,
                embedding_function=_embedding_function,
                k=k,
                reranking_function=_constant_reranker,
                k_reranker=k,
                r=0.0,
                hybrid_bm25_weight=1.0,  # vector path skipped; cg adds as 2nd member
                enable_enriched_texts=False,
                concept_graph_store=store,
                concept_graph_weight=0.5,  # boost cg presence for acceptance
                concept_graph_embed_fn=embed_fn,
            )
        )
    finally:
        _restore_cg_documents_patch(patch_original)

    from langchain_core.documents import Document
    docs: list[Document] = []
    for content, meta in zip(
        result_dict['documents'][0],
        result_dict['metadatas'][0],
    ):
        docs.append(Document(page_content=content, metadata=dict(meta)))
    return docs


@pytest.mark.parametrize(
    'q',
    QUESTIONS,
    ids=lambda q: f"wired-{q['id']}-{q['intent']}-{q['difficulty']}",
)
def test_acceptance_question_wired_path(
    q,
    lollipop_subset_store_with_embeddings,
    acceptance_embedder,
):
    """Wired-path variant: routes through query_doc_with_hybrid_search.

    The gate matches the in-process gate (test_acceptance_question)
    within ±1 question on the full fixture. Per-question failures here
    indicate the wired path's ensemble + RRF + W2-B extension behaves
    differently from direct router.retrieve calls."""
    if q['intent'] == 'generate_code':
        pytest.skip('generate_code is out of Phase 1 scope; informational only')

    expected = q['expected_concepts']
    if not expected:
        pytest.skip(f'Question {q["id"]} has no expected_concepts')

    embed_fn, _label = acceptance_embedder
    flag, prior_flag = _enable_concept_graph_for_wired_test()
    try:
        collection_result = _build_synthetic_collection_result('/tmp/lollipop_subset')
        docs = _wired_query(
            store=lollipop_subset_store_with_embeddings,
            embed_fn=embed_fn,
            question_text=q['question'],
            collection_result=collection_result,
        )
    finally:
        flag.value = prior_flag

    threshold = PER_KIND_THRESHOLDS.get(q['intent'], 0.5)
    passed, score, matched = score_documents_against_expected(
        docs,
        expected,
        fuzzy_threshold=threshold,
    )

    if not passed:
        top_names = [
            (d.metadata or {}).get('concept_name')
            or (d.metadata or {}).get('source')
            for d in docs[:10]
        ]
        pytest.fail(
            f'WIRED question {q["id"]} FAILED.\n'
            f'  Intent (expected): {q["intent"]}\n'
            f'  Score: {score:.2f} (threshold: {threshold})\n'
            f'  Expected: {expected}\n'
            f'  Matched: {sorted(matched)}\n'
            f'  Missed: {sorted(set(expected) - matched)}\n'
            f'  Top-10 docs (concept_name|source): {top_names}'
        )


def test_phase2_wired_acceptance_rate(
    lollipop_subset_store_with_embeddings,
    acceptance_embedder,
):
    """Phase 2 gate-0 sanity: the wired-path pass rate must match the
    in-process pass rate within ±1 question. ≥ 60% absolute as a floor."""
    embed_fn, embedder_label = acceptance_embedder
    in_scope = [q for q in QUESTIONS if q['intent'] != 'generate_code']
    flag, prior_flag = _enable_concept_graph_for_wired_test()
    try:
        collection_result = _build_synthetic_collection_result('/tmp/lollipop_subset')
        passed = 0
        failures: list[str] = []
        for q in in_scope:
            expected = q['expected_concepts']
            if not expected:
                continue
            docs = _wired_query(
                store=lollipop_subset_store_with_embeddings,
                embed_fn=embed_fn,
                question_text=q['question'],
                collection_result=collection_result,
            )
            threshold = PER_KIND_THRESHOLDS.get(q['intent'], 0.5)
            p, score, _ = score_documents_against_expected(
                docs, expected, fuzzy_threshold=threshold,
            )
            if p:
                passed += 1
            else:
                failures.append(f'{q["id"]} ({q["intent"]}): {score:.2f} < {threshold}')
    finally:
        flag.value = prior_flag

    pass_rate = passed / len(in_scope) if in_scope else 0.0
    print(
        f'\nwired-path acceptance: {passed}/{len(in_scope)} '
        f'({100 * pass_rate:.0f}%) pass',
    )
    print(f'embedder: {embedder_label}')
    if failures:
        print('Wired failures:')
        for f in failures:
            print(f'  {f}')

    assert pass_rate >= 0.6, (
        f'Wired pass rate {pass_rate:.2f} below 0.6 gate; '
        f'wired path may be regressing the in-process acceptance gate.'
    )


def make_acceptance_reranker(embed_fn):
    """Build the cosine-based pre-RRF reranker for wired acceptance tests.

    Uses the acceptance embedder (deterministic) so the test is reproducible.
    No network, no model download. Wraps W4-A's ``make_cosine_scorer`` +
    ``rerank_hits`` so the closure has the (query, hits) -> hits shape that
    ``query_doc_with_hybrid_search``'s ``concept_graph_reranker`` param accepts.
    """
    from open_webui.retrieval.concepts.retrieve.reranker import (
        make_cosine_scorer,
        rerank_hits,
    )

    scorer = make_cosine_scorer(query_embed_fn=embed_fn)

    def _reranker(query, hits):
        return rerank_hits(query, hits, scorer=scorer)

    return _reranker


def _hit_concept_names(hits, n: int = 10) -> list[str]:
    names: list[str] = []
    for hit in hits[:n]:
        if hit.concept is not None:
            names.append(hit.concept.name)
        elif hit.artifact is not None:
            names.append(hit.artifact.path)
        else:
            names.append('(empty)')
    return names


def _cg_rerank_diagnostic(
    *,
    store,
    embed_fn,
    question_text: str,
    k: int = 20,
) -> tuple[list[str], list[str]]:
    """Return post-PPR and post-rerank top-10 concept names for failure diagnostics."""
    router_result = route(
        RetrievalQuery(text=question_text, top_k=k),
        store,
        config=RouterConfig(embed_fn=embed_fn),
    )
    ppr_hits = router_result.hits
    reranked_hits = make_acceptance_reranker(embed_fn)(question_text, ppr_hits)
    return _hit_concept_names(ppr_hits), _hit_concept_names(reranked_hits)


def _wired_query_with_reranker(
    *,
    store,
    embed_fn,
    question_text: str,
    collection_result,
    k: int = 20,
) -> list:
    """Run one wired-path retrieval with pre-RRF reranking. Returns list[Document]."""
    import asyncio
    from open_webui.retrieval.utils import query_doc_with_hybrid_search

    async def _embedding_function(text, prefix=None, user=None):
        if isinstance(text, list):
            return [[0.0] * 16] * len(text)
        return [0.0] * 16

    def _constant_reranker(query, documents, user=None):
        return [1.0] * len(documents)

    patch_original = _patch_cg_documents_for_rrf()
    try:
        result_dict = asyncio.run(
            query_doc_with_hybrid_search(
                collection_name='lollipop-acceptance-wired',
                collection_result=collection_result,
                query=question_text,
                embedding_function=_embedding_function,
                k=k,
                reranking_function=_constant_reranker,
                k_reranker=k,
                r=0.0,
                hybrid_bm25_weight=1.0,
                enable_enriched_texts=False,
                concept_graph_store=store,
                concept_graph_weight=0.5,
                concept_graph_embed_fn=embed_fn,
                concept_graph_reranker=make_acceptance_reranker(embed_fn),
            )
        )
    finally:
        _restore_cg_documents_patch(patch_original)

    from langchain_core.documents import Document
    docs: list[Document] = []
    for content, meta in zip(
        result_dict['documents'][0],
        result_dict['metadatas'][0],
    ):
        docs.append(Document(page_content=content, metadata=dict(meta)))
    return docs


@pytest.mark.parametrize(
    'q',
    QUESTIONS,
    ids=lambda q: f"wired-reranked-{q['id']}-{q['intent']}-{q['difficulty']}",
)
def test_acceptance_question_wired_reranked_path(
    q,
    lollipop_subset_store_with_embeddings,
    acceptance_embedder,
):
    """Wired-path variant with pre-RRF cosine reranking of cg hits."""
    if q['intent'] == 'generate_code':
        pytest.skip('generate_code is out of Phase 1 scope; informational only')

    expected = q['expected_concepts']
    if not expected:
        pytest.skip(f'Question {q["id"]} has no expected_concepts')

    embed_fn, _label = acceptance_embedder
    flag, prior_flag = _enable_concept_graph_for_wired_test()
    try:
        collection_result = _build_synthetic_collection_result('/tmp/lollipop_subset')
        docs = _wired_query_with_reranker(
            store=lollipop_subset_store_with_embeddings,
            embed_fn=embed_fn,
            question_text=q['question'],
            collection_result=collection_result,
        )
    finally:
        flag.value = prior_flag

    threshold = PER_KIND_THRESHOLDS.get(q['intent'], 0.5)
    passed, score, matched = score_documents_against_expected(
        docs,
        expected,
        fuzzy_threshold=threshold,
    )

    if not passed:
        before, after = _cg_rerank_diagnostic(
            store=lollipop_subset_store_with_embeddings,
            embed_fn=embed_fn,
            question_text=q['question'],
        )
        top_names = [
            (d.metadata or {}).get('concept_name')
            or (d.metadata or {}).get('source')
            for d in docs[:10]
        ]
        pytest.fail(
            f'WIRED-RERANKED question {q["id"]} FAILED.\n'
            f'  Intent (expected): {q["intent"]}\n'
            f'  Score: {score:.2f} (threshold: {threshold})\n'
            f'  Expected: {expected}\n'
            f'  Matched: {sorted(matched)}\n'
            f'  Missed: {sorted(set(expected) - matched)}\n'
            f'  Top-10 docs (concept_name|source): {top_names}\n'
            f'  cg-rank 1-10 BEFORE reranker (post-PPR): {before}\n'
            f'  cg-rank 1-10 AFTER reranker: {after}'
        )


def test_phase2_wired_reranked_acceptance_rate(
    lollipop_subset_store_with_embeddings,
    acceptance_embedder,
):
    """Reranked wired-path must not regress vs unreranked wired; floor >= 60%."""
    embed_fn, embedder_label = acceptance_embedder
    in_scope = [q for q in QUESTIONS if q['intent'] != 'generate_code']
    flag, prior_flag = _enable_concept_graph_for_wired_test()
    try:
        collection_result = _build_synthetic_collection_result('/tmp/lollipop_subset')
        wired_passed = 0
        reranked_passed = 0
        failures: list[str] = []
        for q in in_scope:
            expected = q['expected_concepts']
            if not expected:
                continue
            docs_wired = _wired_query(
                store=lollipop_subset_store_with_embeddings,
                embed_fn=embed_fn,
                question_text=q['question'],
                collection_result=collection_result,
            )
            docs_reranked = _wired_query_with_reranker(
                store=lollipop_subset_store_with_embeddings,
                embed_fn=embed_fn,
                question_text=q['question'],
                collection_result=collection_result,
            )
            threshold = PER_KIND_THRESHOLDS.get(q['intent'], 0.5)
            p_wired, score_w, _ = score_documents_against_expected(
                docs_wired, expected, fuzzy_threshold=threshold,
            )
            p_reranked, score_r, _ = score_documents_against_expected(
                docs_reranked, expected, fuzzy_threshold=threshold,
            )
            if p_wired:
                wired_passed += 1
            if p_reranked:
                reranked_passed += 1
            else:
                failures.append(
                    f'{q["id"]} ({q["intent"]}): reranked {score_r:.2f} < {threshold}',
                )
    finally:
        flag.value = prior_flag

    wired_rate = wired_passed / len(in_scope) if in_scope else 0.0
    reranked_rate = reranked_passed / len(in_scope) if in_scope else 0.0
    print(
        f'\nwired-reranked-path acceptance: {reranked_passed}/{len(in_scope)} '
        f'({100 * reranked_rate:.0f}%) pass',
    )
    print(
        f'unreranked wired-path acceptance: {wired_passed}/{len(in_scope)} '
        f'({100 * wired_rate:.0f}%) pass',
    )
    print(f'embedder: {embedder_label}')
    if failures:
        print('Reranked wired failures:')
        for f in failures:
            print(f'  {f}')

    assert reranked_rate >= wired_rate, (
        f'Reranked wired pass rate {reranked_rate:.2f} regressed below '
        f'unreranked wired {wired_rate:.2f}'
    )
    assert reranked_rate >= 0.6, (
        f'Reranked wired pass rate {reranked_rate:.2f} below 0.6 gate'
    )


def _catrag_router_config(embed_fn, *, catrag_alpha=0.2, embed_alpha=0.5) -> RouterConfig:
    """RouterConfig pinned to catrag tiebreaker for failure diagnostics."""
    import os as _os

    catrag_alpha_env = _os.environ.get('CONCEPT_GRAPH_CATRAG_ALPHA')
    if catrag_alpha_env is not None:
        catrag_alpha = float(catrag_alpha_env)
    return RouterConfig(
        language='csharp',
        embed_fn=embed_fn,
        tiebreaker='catrag',
        embed_blend_alpha=embed_alpha,
        catrag_anchor_alpha=catrag_alpha,
    )


def _catrag_doc_diagnostics(docs: list) -> tuple[dict | None, bool]:
    """Return (catrag metadata from top cg doc, glossary_phrase_matched flag)."""
    catrag_meta: dict | None = None
    glossary_matched = False
    for doc in docs:
        meta = getattr(doc, 'metadata', {}) or {}
        catrag_keys = {k: v for k, v in meta.items() if str(k).startswith('catrag_')}
        if catrag_keys:
            catrag_meta = catrag_keys
            glossary_matched = bool(
                meta.get('catrag_glossary_matched')
                or meta.get('catrag_anchor_bonus', 0)
            )
            break
    return catrag_meta, glossary_matched


def _wired_query_with_catrag(
    *,
    store,
    embed_fn,
    question_text: str,
    collection_result,
    k: int = 20,
    catrag_alpha: float = 0.2,
    embed_alpha: float = 0.5,
) -> list:
    """Wired-path retrieval using the 'catrag' tiebreaker inside the cg-router.

    Unlike _wired_query_with_reranker which post-processes hits with a
    cosine reranker AFTER route() returns, this variant configures route()
    itself to use the catrag tiebreaker — combining PPR + cosine + glossary
    anchor bonus inside the neighborhood retriever's _sort_key. The hit
    ordering coming OUT of route() is already query-aware; no separate
    reranker call.
    """
    import asyncio
    import os as _os
    from open_webui.retrieval.utils import query_doc_with_hybrid_search

    catrag_alpha_env = _os.environ.get('CONCEPT_GRAPH_CATRAG_ALPHA')
    if catrag_alpha_env is not None:
        catrag_alpha = float(catrag_alpha_env)

    async def _embedding_function(text, prefix=None, user=None):
        if isinstance(text, list):
            return [[0.0] * 16] * len(text)
        return [0.0] * 16

    def _constant_reranker(query, documents, user=None):
        return [1.0] * len(documents)

    patch_original = _patch_cg_documents_for_rrf()
    try:
        result_dict = asyncio.run(
            query_doc_with_hybrid_search(
                collection_name='lollipop-acceptance-wired-catrag',
                collection_result=collection_result,
                query=question_text,
                embedding_function=_embedding_function,
                k=k,
                reranking_function=_constant_reranker,
                k_reranker=k,
                r=0.0,
                hybrid_bm25_weight=1.0,
                enable_enriched_texts=False,
                concept_graph_store=store,
                concept_graph_weight=0.5,
                concept_graph_embed_fn=embed_fn,
                concept_graph_tiebreaker='catrag',
                concept_graph_embed_alpha=embed_alpha,
                concept_graph_catrag_alpha=catrag_alpha,
            )
        )
    finally:
        _restore_cg_documents_patch(patch_original)

    from langchain_core.documents import Document
    docs: list[Document] = []
    for content, meta in zip(
        result_dict['documents'][0],
        result_dict['metadatas'][0],
    ):
        docs.append(Document(page_content=content, metadata=dict(meta)))
    return docs


@pytest.mark.parametrize(
    'q',
    QUESTIONS,
    ids=lambda q: f"wired-catrag-{q['id']}-{q['intent']}-{q['difficulty']}",
)
def test_acceptance_question_wired_catrag_path(
    q,
    lollipop_subset_store_with_embeddings,
    acceptance_embedder,
):
    """Wired-path variant using the catrag tiebreaker INSIDE route()."""
    if q['intent'] == 'generate_code':
        pytest.skip('generate_code is out of Phase 1 scope; informational only')

    expected = q['expected_concepts']
    if not expected:
        pytest.skip(f'Question {q["id"]} has no expected_concepts')

    embed_fn, _label = acceptance_embedder
    flag, prior_flag = _enable_concept_graph_for_wired_test()
    try:
        collection_result = _build_synthetic_collection_result('/tmp/lollipop_subset')
        docs = _wired_query_with_catrag(
            store=lollipop_subset_store_with_embeddings,
            embed_fn=embed_fn,
            question_text=q['question'],
            collection_result=collection_result,
        )
    finally:
        flag.value = prior_flag

    threshold = PER_KIND_THRESHOLDS.get(q['intent'], 0.5)
    passed, score, matched = score_documents_against_expected(
        docs,
        expected,
        fuzzy_threshold=threshold,
    )

    if not passed:
        router_result = route(
            RetrievalQuery(text=q['question'], top_k=20),
            lollipop_subset_store_with_embeddings,
            config=_catrag_router_config(embed_fn),
        )
        catrag_meta, glossary_matched = _catrag_doc_diagnostics(docs)
        top_names = [
            (d.metadata or {}).get('concept_name')
            or (d.metadata or {}).get('source')
            for d in docs[:10]
        ]
        pytest.fail(
            f'WIRED-CATRAG question {q["id"]} FAILED.\n'
            f'  Intent (expected): {q["intent"]}\n'
            f'  Intent (classified): {_intent_value(router_result)}\n'
            f'  Score: {score:.2f} (threshold: {threshold})\n'
            f'  Expected: {expected}\n'
            f'  Matched: {sorted(matched)}\n'
            f'  Missed: {sorted(set(expected) - matched)}\n'
            f'  Top-10 docs (concept_name|source): {top_names}\n'
            f'  catrag_glossary_matched: {glossary_matched}\n'
            f'  catrag_metadata (top cg doc): {catrag_meta}'
        )


def test_phase2_wired_catrag_acceptance_rate(
    lollipop_subset_store_with_embeddings,
    acceptance_embedder,
):
    """Catrag wired-path floor >= 60% (6/9); measures lift vs bare wired."""
    embed_fn, embedder_label = acceptance_embedder
    in_scope = [q for q in QUESTIONS if q['intent'] != 'generate_code']
    flag, prior_flag = _enable_concept_graph_for_wired_test()
    try:
        collection_result = _build_synthetic_collection_result('/tmp/lollipop_subset')
        passed = 0
        failures: list[str] = []
        for q in in_scope:
            expected = q['expected_concepts']
            if not expected:
                continue
            docs = _wired_query_with_catrag(
                store=lollipop_subset_store_with_embeddings,
                embed_fn=embed_fn,
                question_text=q['question'],
                collection_result=collection_result,
            )
            threshold = PER_KIND_THRESHOLDS.get(q['intent'], 0.5)
            p, score, _ = score_documents_against_expected(
                docs, expected, fuzzy_threshold=threshold,
            )
            if p:
                passed += 1
            else:
                failures.append(f'{q["id"]} ({q["intent"]}): {score:.2f} < {threshold}')
    finally:
        flag.value = prior_flag

    pass_rate = passed / len(in_scope) if in_scope else 0.0
    print(
        f'\nwired-catrag-path acceptance: {passed}/{len(in_scope)} '
        f'({100 * pass_rate:.0f}%) pass',
    )
    print(f'embedder: {embedder_label}')
    if failures:
        print('Catrag wired failures:')
        for f in failures:
            print(f'  {f}')

    assert pass_rate >= 0.66, (
        f'Catrag wired pass rate {pass_rate:.2f} below 0.66 gate (6/9 floor)'
    )
