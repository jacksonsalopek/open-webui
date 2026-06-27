"""Tests for ``open_webui.retrieval.concepts.extraction.glossary``."""

from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest

from open_webui.retrieval.concepts.extraction.glossary import (
    Glossary,
    PhraseConcept,
)
from open_webui.retrieval.concepts.schema import ConceptKind, concept_from_dict, concept_to_dict

_TS = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
_TS2 = datetime(2025, 6, 2, 8, 30, 0, tzinfo=timezone.utc)

_SEED_PHRASE_COUNT = 20

_W10B_NAMES: tuple[str, ...] = (
    'invoke-pattern',
    'selection-pattern',
    'value-pattern',
    'text-pattern',
    'toggle-pattern',
    'expand-collapse-pattern',
    'range-value-pattern',
    'scroll-pattern',
    'windows-hook',
    'wh-keyboard-ll',
    'wh-mouse-ll',
    'winevent-hook',
    'set-windows-hook-ex',
    'low-level-hook',
    'uiautomation-element',
    'iaccessible',
    'automation-id',
    'accessibility-tree',
    'system-prompt',
    'tool-call',
    'prompt-injection',
    'guardrail-policy',
    'safety-guardrail',
    'selection-gesture',
    'clipboard-snapshot',
    'llm-completion',
    'global-hook-service',
    'chat-response',
)

# WinUI 3 / .NET 10 modern-desktop vocabulary added 2026-06-26 during the
# zap cross-corpus probe. Generic terms-of-art, not zap-specific; see
# CONCEPT_GRAPH_PHASE1.md § "Post-closure cross-corpus probe" for rationale.
_ZAP_NAMES: tuple[str, ...] = (
    'mica-backdrop',
    'system-backdrop',
    'acrylic-backdrop',
    'custom-title-bar',
    'fluent-design',
    'immersive-dark-mode',
    'element-theme',
    'application-data',
    'winui',
    'single-instance',
    'named-mutex',
    'named-pipe',
    'protocol-activation',
    'webp-format',
    'jpeg-format',
    'png-format',
    'libvips',
    'app-installer',
    'msix-package',
    'explorer-command',
    'shell-context-menu',
    'command-palette',
)

_KEBAB_CASE = re.compile(r'^[a-z][a-z0-9-]*[a-z0-9]$')


def _normalize_surface_form(form: str) -> str:
    return re.sub(r'[ _-]+', ' ', form.strip().lower())


@pytest.fixture
def default_glossary() -> Glossary:
    return Glossary.default()


def test_load_default_glossary_has_20_phrases(default_glossary: Glossary) -> None:
    assert (
        len(default_glossary.phrases)
        == _SEED_PHRASE_COUNT + len(_W10B_NAMES) + len(_ZAP_NAMES)
    )


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
    assert (
        len(concepts) == _SEED_PHRASE_COUNT + len(_W10B_NAMES) + len(_ZAP_NAMES)
    )
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
    assert (
        len(merged.phrases)
        == _SEED_PHRASE_COUNT + len(_W10B_NAMES) + len(_ZAP_NAMES)
    )


def test_match_hits_sorted_by_start(default_glossary: Glossary) -> None:
    hits = default_glossary.match(
        'race condition during garbage collector startup on the hot path',
    )
    starts = [hit.span[0] for hit in hits]
    assert starts == sorted(starts)
    assert len(hits) >= 3


def test_empty_text_returns_empty_list(default_glossary: Glossary) -> None:
    assert default_glossary.match('') == []


def test_w10b_block_parses(default_glossary: Glossary) -> None:
    names = {phrase.name for phrase in default_glossary.phrases}
    present = [name for name in _W10B_NAMES if name in names]
    assert len(present) >= 25


def test_w10b_aliases_non_empty(default_glossary: Glossary) -> None:
    by_name = {phrase.name: phrase for phrase in default_glossary.phrases}
    for name in _W10B_NAMES:
        phrase = by_name[name]
        assert len(phrase.surface_forms) >= 2, f'{name} needs at least 2 surface forms'


def test_w10b_no_alias_collisions_with_existing(default_glossary: Glossary) -> None:
    w10b_set = set(_W10B_NAMES)
    pre_existing: set[str] = set()
    w10b_aliases: set[str] = set()

    for phrase in default_glossary.phrases:
        normalized = {_normalize_surface_form(form) for form in phrase.surface_forms}
        if phrase.name in w10b_set:
            w10b_aliases.update(normalized)
        else:
            pre_existing.update(normalized)

    collisions = w10b_aliases & pre_existing
    assert not collisions, f'W10-B aliases collide with pre-existing forms: {sorted(collisions)}'


def test_w10b_names_are_kebab_case() -> None:
    for name in _W10B_NAMES:
        assert _KEBAB_CASE.match(name), f'{name!r} is not kebab-case'


def test_zap_block_parses(default_glossary: Glossary) -> None:
    names = {phrase.name for phrase in default_glossary.phrases}
    present = [name for name in _ZAP_NAMES if name in names]
    assert len(present) == len(_ZAP_NAMES), (
        f'Missing zap names: {sorted(set(_ZAP_NAMES) - names)}'
    )


def test_zap_aliases_non_empty(default_glossary: Glossary) -> None:
    by_name = {phrase.name: phrase for phrase in default_glossary.phrases}
    for name in _ZAP_NAMES:
        phrase = by_name[name]
        assert len(phrase.surface_forms) >= 2, f'{name} needs at least 2 surface forms'


def test_zap_no_alias_collisions_with_existing(default_glossary: Glossary) -> None:
    zap_set = set(_ZAP_NAMES)
    pre_existing: set[str] = set()
    zap_aliases: set[str] = set()

    for phrase in default_glossary.phrases:
        normalized = {_normalize_surface_form(form) for form in phrase.surface_forms}
        if phrase.name in zap_set:
            zap_aliases.update(normalized)
        else:
            pre_existing.update(normalized)

    collisions = zap_aliases & pre_existing
    assert not collisions, f'Zap aliases collide with pre-existing forms: {sorted(collisions)}'


def test_zap_names_are_kebab_case() -> None:
    for name in _ZAP_NAMES:
        assert _KEBAB_CASE.match(name), f'{name!r} is not kebab-case'
