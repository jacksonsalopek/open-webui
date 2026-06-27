"""Phase 2 W3 wired-path acceptance harness — Zap-WinUI corpus.

Runs each of the 10 Zap acceptance questions through the production
``query_doc_with_hybrid_search(...)`` retrieval path with the
concept-graph store wired in (W2 gate-0 hooks). Mirrors
test_acceptance_lollipop.py::test_acceptance_question_wired_path
but on the Zap corpus + zap_v1.yaml + /tmp/zap_subset.

Reuses W3-A's score_documents_against_expected and
_build_synthetic_collection_result so the scoring + chunk-synthesis
semantics are identical across corpora.

Aggregate gate: ≥ 50% (5/10) — matches the Phase 1.5 cross-corpus
probe baseline of 6/10. Below 5/10 indicates a Phase 2 regression."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytest.importorskip(
    'open_webui.retrieval.concepts.retrieve.router',
    reason='Step 9 router not implemented yet — acceptance harness skipped',
)

# Reuse W3-A's helpers — exported from test_acceptance_lollipop.
# If W3-A renamed these, follow whatever the actual exported names are
# and document in the report.
from open_webui.retrieval.concepts.retrieve.base import RetrievalQuery
from open_webui.retrieval.concepts.retrieve.router import route

from open_webui.test.apps.webui.concepts.test_acceptance_lollipop import (
    PER_KIND_THRESHOLDS,
    score_documents_against_expected,
    _build_synthetic_collection_result,
    _wired_query,
    _enable_concept_graph_for_wired_test,
    make_acceptance_reranker,
    _cg_rerank_diagnostic,
    _catrag_router_config,
    _catrag_doc_diagnostics,
    _intent_value,
)


def _zap_yaml_path() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / 'retrieval' / 'concepts' / 'acceptance' / 'zap_v1.yaml'
        if candidate.is_file():
            return candidate
    raise FileNotFoundError('zap_v1.yaml not found relative to test module')


def _load_zap_questions() -> list[dict]:
    with open(_zap_yaml_path()) as f:
        data = yaml.safe_load(f)
    return data['questions']


QUESTIONS = _load_zap_questions()


@pytest.mark.parametrize(
    'q',
    QUESTIONS,
    ids=lambda q: f"zap-wired-{q['id']}-{q['intent']}-{q['difficulty']}",
)
def test_acceptance_question_zap_wired(
    q,
    zap_subset_store_with_embeddings,
    acceptance_embedder,
):
    """One Zap question, wired-path. Passes iff per-intent score >= threshold.

    Routes through query_doc_with_hybrid_search with hybrid_bm25_weight=1.0
    and concept_graph_store set. See test_acceptance_lollipop's wired-path
    docstring for the design rationale."""
    if q['intent'] == 'generate_code':
        pytest.skip('generate_code is out of Phase 1 scope; informational only')

    expected = q['expected_concepts']
    if not expected:
        pytest.skip(f'Question {q["id"]} has no expected_concepts')

    embed_fn, _label = acceptance_embedder
    flag, prior_flag = _enable_concept_graph_for_wired_test()
    try:
        collection_result = _build_synthetic_collection_result('/tmp/zap_subset')
        docs = _wired_query(
            store=zap_subset_store_with_embeddings,
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
            f'WIRED Zap question {q["id"]} FAILED.\n'
            f'  Intent (expected): {q["intent"]}\n'
            f'  Score: {score:.2f} (threshold: {threshold})\n'
            f'  Expected: {expected}\n'
            f'  Matched: {sorted(matched)}\n'
            f'  Missed: {sorted(set(expected) - matched)}\n'
            f'  Top-10 docs (concept_name|source): {top_names}'
        )


def test_phase2_zap_wired_acceptance_rate(
    zap_subset_store_with_embeddings,
    acceptance_embedder,
):
    """Aggregate Zap wired-path gate.

    Floor: 5/10 (50%). Reports drift from the Phase 1.5 cross-corpus
    probe baseline of 6/10 as diagnostic. Below 50% fails the gate."""
    embed_fn, embedder_label = acceptance_embedder
    in_scope = [q for q in QUESTIONS if q['intent'] != 'generate_code']
    flag, prior_flag = _enable_concept_graph_for_wired_test()
    try:
        collection_result = _build_synthetic_collection_result('/tmp/zap_subset')
        passed = 0
        failures: list[str] = []
        for q in in_scope:
            expected = q['expected_concepts']
            if not expected:
                continue
            docs = _wired_query(
                store=zap_subset_store_with_embeddings,
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
        f'\nzap-wired-path acceptance: {passed}/{len(in_scope)} '
        f'({100 * pass_rate:.0f}%) pass',
    )
    print(f'embedder: {embedder_label}')
    if failures:
        print('Wired failures:')
        for f in failures:
            print(f'  {f}')

    if passed < 6:
        print(
            f'\nNOTE: Zap wired-path is {passed}/10; Phase 1.5 in-process baseline was 6/10. '
            f'Drift from baseline = {passed - 6}.'
        )

    assert pass_rate >= 0.5, (
        f'Zap wired pass rate {pass_rate:.2f} below 0.5 gate; '
        f'wired path may be regressing the in-process acceptance gate.'
    )


def _wired_query_with_reranker(
    *,
    store,
    embed_fn,
    question_text: str,
    collection_result,
    k: int = 20,
) -> list:
    """Zap wired-path retrieval with pre-RRF cosine reranking."""
    import asyncio
    from open_webui.retrieval.utils import query_doc_with_hybrid_search

    async def _embedding_function(text, prefix=None, user=None):
        if isinstance(text, list):
            return [[0.0] * 16] * len(text)
        return [0.0] * 16

    def _constant_reranker(query, documents, user=None):
        return [1.0] * len(documents)

    from open_webui.test.apps.webui.concepts.test_acceptance_lollipop import (
        _patch_cg_documents_for_rrf,
        _restore_cg_documents_patch,
    )

    patch_original = _patch_cg_documents_for_rrf()
    try:
        result_dict = asyncio.run(
            query_doc_with_hybrid_search(
                collection_name='zap-acceptance-wired',
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
    ids=lambda q: f"zap-wired-reranked-{q['id']}-{q['intent']}-{q['difficulty']}",
)
def test_acceptance_question_zap_wired_reranked(
    q,
    zap_subset_store_with_embeddings,
    acceptance_embedder,
):
    """One Zap question, wired-path with pre-RRF reranking."""
    if q['intent'] == 'generate_code':
        pytest.skip('generate_code is out of Phase 1 scope; informational only')

    expected = q['expected_concepts']
    if not expected:
        pytest.skip(f'Question {q["id"]} has no expected_concepts')

    embed_fn, _label = acceptance_embedder
    flag, prior_flag = _enable_concept_graph_for_wired_test()
    try:
        collection_result = _build_synthetic_collection_result('/tmp/zap_subset')
        docs = _wired_query_with_reranker(
            store=zap_subset_store_with_embeddings,
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
            store=zap_subset_store_with_embeddings,
            embed_fn=embed_fn,
            question_text=q['question'],
        )
        top_names = [
            (d.metadata or {}).get('concept_name')
            or (d.metadata or {}).get('source')
            for d in docs[:10]
        ]
        pytest.fail(
            f'WIRED-RERANKED Zap question {q["id"]} FAILED.\n'
            f'  Intent (expected): {q["intent"]}\n'
            f'  Score: {score:.2f} (threshold: {threshold})\n'
            f'  Expected: {expected}\n'
            f'  Matched: {sorted(matched)}\n'
            f'  Missed: {sorted(set(expected) - matched)}\n'
            f'  Top-10 docs (concept_name|source): {top_names}\n'
            f'  cg-rank 1-10 BEFORE reranker (post-PPR): {before}\n'
            f'  cg-rank 1-10 AFTER reranker: {after}'
        )


def test_phase2_zap_wired_reranked_acceptance_rate(
    zap_subset_store_with_embeddings,
    acceptance_embedder,
):
    """Reranked Zap wired-path must not regress vs unreranked; floor >= 50%."""
    embed_fn, embedder_label = acceptance_embedder
    in_scope = [q for q in QUESTIONS if q['intent'] != 'generate_code']
    flag, prior_flag = _enable_concept_graph_for_wired_test()
    try:
        collection_result = _build_synthetic_collection_result('/tmp/zap_subset')
        wired_passed = 0
        reranked_passed = 0
        failures: list[str] = []
        for q in in_scope:
            expected = q['expected_concepts']
            if not expected:
                continue
            docs_wired = _wired_query(
                store=zap_subset_store_with_embeddings,
                embed_fn=embed_fn,
                question_text=q['question'],
                collection_result=collection_result,
            )
            docs_reranked = _wired_query_with_reranker(
                store=zap_subset_store_with_embeddings,
                embed_fn=embed_fn,
                question_text=q['question'],
                collection_result=collection_result,
            )
            threshold = PER_KIND_THRESHOLDS.get(q['intent'], 0.5)
            p_wired, _, _ = score_documents_against_expected(
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
        f'\nzap-wired-reranked-path acceptance: {reranked_passed}/{len(in_scope)} '
        f'({100 * reranked_rate:.0f}%) pass',
    )
    print(
        f'unreranked zap-wired-path acceptance: {wired_passed}/{len(in_scope)} '
        f'({100 * wired_rate:.0f}%) pass',
    )
    print(f'embedder: {embedder_label}')
    if failures:
        print('Reranked Zap wired failures:')
        for f in failures:
            print(f'  {f}')

    assert reranked_rate >= wired_rate, (
        f'Reranked Zap wired pass rate {reranked_rate:.2f} regressed below '
        f'unreranked wired {wired_rate:.2f}'
    )
    assert reranked_rate >= 0.5, (
        f'Reranked Zap wired pass rate {reranked_rate:.2f} below 0.5 gate'
    )


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
    """Zap wired-path retrieval using the 'catrag' tiebreaker inside route()."""
    import asyncio
    import os as _os
    from open_webui.retrieval.utils import query_doc_with_hybrid_search

    from open_webui.test.apps.webui.concepts.test_acceptance_lollipop import (
        _patch_cg_documents_for_rrf,
        _restore_cg_documents_patch,
    )

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
                collection_name='zap-acceptance-wired-catrag',
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
    ids=lambda q: f"zap-wired-catrag-{q['id']}-{q['intent']}-{q['difficulty']}",
)
def test_acceptance_question_zap_wired_catrag(
    q,
    zap_subset_store_with_embeddings,
    acceptance_embedder,
):
    """One Zap question, wired-path with catrag tiebreaker inside route()."""
    if q['intent'] == 'generate_code':
        pytest.skip('generate_code is out of Phase 1 scope; informational only')

    expected = q['expected_concepts']
    if not expected:
        pytest.skip(f'Question {q["id"]} has no expected_concepts')

    embed_fn, _label = acceptance_embedder
    flag, prior_flag = _enable_concept_graph_for_wired_test()
    try:
        collection_result = _build_synthetic_collection_result('/tmp/zap_subset')
        docs = _wired_query_with_catrag(
            store=zap_subset_store_with_embeddings,
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
            zap_subset_store_with_embeddings,
            config=_catrag_router_config(embed_fn),
        )
        catrag_meta, glossary_matched = _catrag_doc_diagnostics(docs)
        top_names = [
            (d.metadata or {}).get('concept_name')
            or (d.metadata or {}).get('source')
            for d in docs[:10]
        ]
        pytest.fail(
            f'WIRED-CATRAG Zap question {q["id"]} FAILED.\n'
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


def test_phase2_zap_wired_catrag_acceptance_rate(
    zap_subset_store_with_embeddings,
    acceptance_embedder,
):
    """Catrag Zap wired-path floor >= 50% (5/10)."""
    embed_fn, embedder_label = acceptance_embedder
    in_scope = [q for q in QUESTIONS if q['intent'] != 'generate_code']
    flag, prior_flag = _enable_concept_graph_for_wired_test()
    try:
        collection_result = _build_synthetic_collection_result('/tmp/zap_subset')
        passed = 0
        failures: list[str] = []
        for q in in_scope:
            expected = q['expected_concepts']
            if not expected:
                continue
            docs = _wired_query_with_catrag(
                store=zap_subset_store_with_embeddings,
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
        f'\nzap-wired-catrag-path acceptance: {passed}/{len(in_scope)} '
        f'({100 * pass_rate:.0f}%) pass',
    )
    print(f'embedder: {embedder_label}')
    if failures:
        print('Catrag Zap wired failures:')
        for f in failures:
            print(f'  {f}')

    assert pass_rate >= 0.5, (
        f'Catrag Zap wired pass rate {pass_rate:.2f} below 0.5 gate (5/10 floor)'
    )
