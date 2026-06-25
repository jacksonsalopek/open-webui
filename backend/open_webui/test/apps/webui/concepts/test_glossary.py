"""Tests for ``open_webui.retrieval.concepts.extraction.glossary``."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from open_webui.retrieval.concepts.extraction.glossary import (
    Glossary,
    PhraseConcept,
)
from open_webui.retrieval.concepts.schema import ConceptKind, concept_from_dict, concept_to_dict

_TS = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
_TS2 = datetime(2025, 6, 2, 8, 30, 0, tzinfo=timezone.utc)


@pytest.fixture
def default_glossary() -> Glossary:
    return Glossary.default()


def test_load_default_glossary_has_20_phrases(default_glossary: Glossary) -> None:
    assert len(default_glossary.phrases) == 20


def test_each_default_phrase_has_required_fields(default_glossary: Glossary) -> None:
    for phrase in default_glossary.phrases:
        assert phrase.name
        assert phrase.surface_forms
        assert phrase.definition


def test_match_view_model_with_dash(default_glossary: Glossary) -> None:
    hits = default_glossary.match('the view-model exposes commands')
    assert len(hits) == 1
    assert hits[0].phrase.name == 'view-model'


def test_match_view_model_with_space(default_glossary: Glossary) -> None:
    hits = default_glossary.match('the view model exposes commands')
    assert len(hits) == 1
    assert hits[0].phrase.name == 'view-model'


def test_match_case_insensitive(default_glossary: Glossary) -> None:
    hits = default_glossary.match('Race Condition')
    assert len(hits) == 1
    assert hits[0].phrase.name == 'race-condition'


def test_longest_match_wins(default_glossary: Glossary) -> None:
    hits = default_glossary.match('race condition during garbage collector startup')
    names = {hit.phrase.name for hit in hits}
    assert names == {'race-condition', 'garbage-collector'}


def test_no_match_inside_identifier(default_glossary: Glossary) -> None:
    hits = default_glossary.match('the ViewModelFactory class')
    assert hits == []


def test_no_match_inside_snake_identifier(default_glossary: Glossary) -> None:
    hits = default_glossary.match('the view_model_factory function')
    assert hits == []


def test_to_concepts_emits_phrase_kind(default_glossary: Glossary) -> None:
    concepts = default_glossary.to_concepts(
        language_hint='csharp',
        first_seen_at=_TS,
        last_seen_at=_TS2,
    )
    assert len(concepts) == 20
    for concept, phrase in zip(concepts, default_glossary.phrases, strict=True):
        assert concept.kind == ConceptKind.PHRASE
        assert concept.definition is not None
        assert concept.definition == phrase.definition


def test_to_concepts_passes_schema_invariant(default_glossary: Glossary) -> None:
    concepts = default_glossary.to_concepts(
        first_seen_at=_TS,
        last_seen_at=_TS2,
    )
    for concept in concepts:
        restored = concept_from_dict(concept_to_dict(concept))
        assert restored == concept


def test_merge_later_overrides_earlier(default_glossary: Glossary) -> None:
    override = Glossary(
        [
            PhraseConcept(
                name='race-condition',
                surface_forms=('race condition',),
                definition='Overridden definition for merge test.',
                tags=('test',),
            ),
        ],
    )
    merged = default_glossary.merge(override)
    race = next(p for p in merged.phrases if p.name == 'race-condition')
    assert race.definition == 'Overridden definition for merge test.'
    assert len(merged.phrases) == 20


def test_match_hits_sorted_by_start(default_glossary: Glossary) -> None:
    hits = default_glossary.match(
        'race condition during garbage collector startup on the hot path',
    )
    starts = [hit.span[0] for hit in hits]
    assert starts == sorted(starts)
    assert len(hits) >= 3


def test_empty_text_returns_empty_list(default_glossary: Glossary) -> None:
    assert default_glossary.match('') == []
