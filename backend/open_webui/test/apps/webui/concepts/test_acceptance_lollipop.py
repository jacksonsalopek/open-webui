"""Phase 1 acceptance harness — 10 curated Lollipop developer questions."""

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

    Empty ``expected_concepts`` raises ValueError (a question without expected
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


@pytest.mark.parametrize(
    'q',
    QUESTIONS,
    ids=lambda q: f"{q['id']}-{q['intent']}-{q['difficulty']}",
)
def test_acceptance_question(q, lollipop_subset_store):
    """One acceptance question. Pass iff score >= per-question threshold."""
    if q['intent'] == 'generate_code':
        pytest.skip('generate_code is out of Phase 1 scope; informational only')

    expected = q['expected_concepts']
    if not expected:
        pytest.skip(f'Question {q["id"]} has no expected_concepts')

    config = RouterConfig(language='csharp')
    query = RetrievalQuery(text=q['question'], top_k=20)
    result = route(query, lollipop_subset_store, config=config)

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


def test_phase1_acceptance_rate(lollipop_subset_store):
    """Phase 1 acceptance gate: ≥ 60% of in-scope questions must pass."""
    config = RouterConfig(language='csharp')
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
            lollipop_subset_store,
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
    print(f'\nPhase 1 acceptance: {passed}/{len(in_scope)} = {100 * pass_rate:.0f}%')
    if failures:
        print('Failures:')
        for failure in failures:
            print(f'  {failure}')
    print('Per-retriever:')
    for retriever, counts in sorted(by_retriever.items()):
        print(f'  {retriever}: {counts["passed"]}/{counts["total"]}')

    assert pass_rate >= 0.6, f'Pass rate {pass_rate:.2f} below 0.6 gate'
