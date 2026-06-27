"""Phase 2 W3 acceptance harness — 10 curated Zap-WinUI developer questions.

Cross-corpus baseline: mirrors test_acceptance_lollipop.py structure but
targets the WinUI 3 starter template at /tmp/zap_subset. Same scorer, same
per-intent thresholds. Aggregate gate is 50% (5/10) per the cross-corpus
probe finding in CONCEPT_GRAPH_PHASE1.md §"Post-closure cross-corpus probe".
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from open_webui.retrieval.concepts.retrieve.base import RetrievalQuery

pytest.importorskip(
    'open_webui.retrieval.concepts.retrieve.router',
    reason='Step 9 router not implemented yet — acceptance harness skipped',
)
from open_webui.retrieval.concepts.retrieve.router import RouterConfig, route

from open_webui.test.apps.webui.concepts.test_acceptance_lollipop import (
    score_hit_set_against_expected,
    _router_config,
    _intent_value,
    _hit_display_names,
)

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
        candidate = parent / 'retrieval' / 'concepts' / 'acceptance' / 'zap_v1.yaml'
        if candidate.is_file():
            return candidate
    raise FileNotFoundError('zap_v1.yaml not found relative to test module')


def _load_questions() -> list[dict]:
    with open(_acceptance_yaml_path()) as f:
        data = yaml.safe_load(f)
    return data['questions']


QUESTIONS = _load_questions()


@pytest.mark.parametrize(
    'q',
    QUESTIONS,
    ids=lambda q: f"zap-{q['id']}-{q['intent']}-{q['difficulty']}",
)
def test_acceptance_question_zap(q, zap_subset_store_with_embeddings, acceptance_embedder):
    """One Zap acceptance question. Pass iff score >= per-question threshold."""
    if q['intent'] == 'generate_code':
        pytest.skip('generate_code is out of Phase 1 scope; informational only')

    expected = q['expected_concepts']
    if not expected:
        pytest.skip(f'Question {q["id"]} has no expected_concepts')

    embed_fn, _label = acceptance_embedder
    config = _router_config(embed_fn)
    query = RetrievalQuery(text=q['question'], top_k=20)
    result = route(query, zap_subset_store_with_embeddings, config=config)

    threshold = PER_KIND_THRESHOLDS.get(q['intent'], 0.5)
    passed, score, matched = score_hit_set_against_expected(
        result.hits,
        expected,
        fuzzy_threshold=threshold,
    )

    if not passed:
        pytest.fail(
            f'Zap question {q["id"]} FAILED.\n'
            f'  Intent classified: {_intent_value(result)}\n'
            f'  Retriever used: {result.retriever_used}\n'
            f'  Score: {score:.2f} (threshold: {threshold})\n'
            f'  Expected: {expected}\n'
            f'  Matched: {sorted(matched)}\n'
            f'  Missed: {sorted(set(expected) - matched)}\n'
            f'  Top-10 retrieved: {_hit_display_names(result.hits)}'
        )


def test_zap_acceptance_rate(
    zap_subset_store_with_embeddings,
    acceptance_embedder,
):
    """Zap acceptance gate: ≥ 50% of in-scope questions must pass.

    Phase 1.5 cross-corpus probe measured 6/10 on Zap with default tuning.
    Phase 2 W3 baseline gate is therefore 5/10 (50%); we record any drift
    from the 6/10 baseline as a diagnostic but only fail below 50%."""
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
            zap_subset_store_with_embeddings,
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

    pass_rate = passed / len(in_scope) if in_scope else 0.0
    retriever_parts = [
        f'{name}: {counts["passed"]}'
        for name, counts in sorted(by_retriever.items())
    ]
    retriever_summary = ', '.join(retriever_parts)
    print(
        f'\nzap-acceptance: {passed}/{len(in_scope)} ({100 * pass_rate:.0f}%) pass '
        f'[{retriever_summary}]',
    )
    print(f'embedder: {embedder_label}')
    if failures:
        print('Failures:')
        for failure in failures:
            print(f'  {failure}')

    if passed < 6:
        print(
            f'\nNOTE: Zap acceptance is {passed}/10; Phase 1.5 probe baseline was 6/10. '
            f'Drift from baseline = {passed - 6}.'
        )

    assert pass_rate >= 0.5, f'Zap pass rate {pass_rate:.2f} below 0.5 gate'
