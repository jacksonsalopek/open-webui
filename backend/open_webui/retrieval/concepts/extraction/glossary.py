"""YAML-backed glossary loader and phrase matcher for concept extraction.

Phrase concepts (terms of art) are curated in glossary YAML files and matched
against free text before atomic identifier tokenization runs. The bundled
``default.yaml`` ships under ``concepts/glossary/``; per-project glossaries
layer on top via ``Glossary.from_paths`` or ``Glossary.merge``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from open_webui.retrieval.concepts.schema import Concept, ConceptKind

log = logging.getLogger(__name__)

_BOUNDARY_CHAR = re.compile(r'[A-Za-z0-9_]')
_DEFAULT_YAML = Path(__file__).resolve().parent.parent / 'glossary' / 'default.yaml'


@dataclass(frozen=True, slots=True)
class PhraseConcept:
    """A phrase-level concept whose meaning is NOT just composition
    of its parts (terms of art). Lives in glossary YAML as the
    curated source-of-truth for ``Concept(kind=PHRASE)`` records."""

    name: str
    surface_forms: tuple[str, ...]
    definition: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PhraseHit:
    """One match of a glossary phrase in a source text."""

    phrase: PhraseConcept
    span: tuple[int, int]
    matched_surface: str


@dataclass(frozen=True, slots=True)
class _PatternEntry:
    regex: str
    phrase: PhraseConcept
    surface_form: str


class Glossary:
    """Loadable, mergeable glossary of phrase concepts.

    Loading from YAML, merging multiple sources, matching against
    free text, and emitting Concept records for the graph all live here.
    """

    def __init__(self, phrases: Sequence[PhraseConcept]) -> None:
        self._phrases = tuple(phrases)
        self._entries = _build_pattern_entries(self._phrases)
        self._regex = _compile_match_regex(self._entries)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Glossary:
        yaml_path = Path(path)
        log.debug('loading glossary from %s', yaml_path)
        with yaml_path.open(encoding='utf-8') as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise ValueError(f'glossary YAML root must be a mapping: {yaml_path}')
        return cls(_parse_phrases(data.get('phrases', []), source=yaml_path))

    @classmethod
    def from_paths(cls, paths: Iterable[str | Path]) -> Glossary:
        merged = cls(())
        for path in paths:
            merged = merged.merge(cls.from_yaml(path))
        return merged

    @classmethod
    def default(cls) -> Glossary:
        """Load the bundled default.yaml shipped beside this module."""
        return cls.from_yaml(_DEFAULT_YAML)

    @property
    def phrases(self) -> tuple[PhraseConcept, ...]:
        return self._phrases

    def match(self, text: str) -> list[PhraseHit]:
        """Find all non-overlapping phrase matches in ``text``."""
        if not text:
            return []

        candidates: list[PhraseHit] = []
        for match in self._regex.finditer(text):
            entry = self._entries[match.lastindex - 1]  # type: ignore[operator]
            start, end = match.span(match.lastindex)
            matched_surface = text[start:end]

            if start > 0 and _BOUNDARY_CHAR.match(text[start - 1]):
                continue
            if end < len(text) and _BOUNDARY_CHAR.match(text[end]):
                continue

            candidates.append(
                PhraseHit(
                    phrase=entry.phrase,
                    span=(start, end),
                    matched_surface=matched_surface,
                ),
            )

        return _dedupe_overlaps(candidates)

    def to_concepts(
        self,
        *,
        language_hint: str | None = None,
        first_seen_at: datetime,
        last_seen_at: datetime,
    ) -> list[Concept]:
        """Emit one ``Concept(kind=PHRASE)`` per entry in the glossary."""
        concepts: list[Concept] = []
        for index, phrase in enumerate(self._phrases, start=1):
            concepts.append(
                Concept(
                    id=index,
                    name=phrase.name,
                    kind=ConceptKind.PHRASE,
                    first_seen_at=first_seen_at,
                    last_seen_at=last_seen_at,
                    centrality_score=None,
                    embedding=None,
                    definition=phrase.definition,
                    language_hint=language_hint,
                    original_tokens=phrase.surface_forms,
                ),
            )
        return concepts

    def merge(self, other: Glossary) -> Glossary:
        """Return a new Glossary; entries in ``other`` override entries
        with the same ``name`` in ``self``. Used by ``from_paths``."""
        by_name = {phrase.name: phrase for phrase in self._phrases}
        for phrase in other.phrases:
            by_name[phrase.name] = phrase
        return Glossary(list(by_name.values()))


def _parse_phrases(raw: object, *, source: Path) -> list[PhraseConcept]:
    if not isinstance(raw, list):
        raise ValueError(f'glossary "phrases" must be a list: {source}')

    phrases: list[PhraseConcept] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f'phrase entry {index} must be a mapping: {source}')

        name = entry.get('name')
        surface_forms = entry.get('surface_forms')
        definition = entry.get('definition')

        if not isinstance(name, str) or not name.strip():
            raise ValueError(f'phrase entry {index} missing non-empty name: {source}')
        if not isinstance(surface_forms, list) or not surface_forms:
            raise ValueError(f'phrase {name!r} missing surface_forms: {source}')
        if not isinstance(definition, str) or not definition.strip():
            raise ValueError(f'phrase {name!r} missing definition: {source}')

        tags_raw = entry.get('tags', [])
        if tags_raw is None:
            tags_raw = []
        if not isinstance(tags_raw, list):
            raise ValueError(f'phrase {name!r} tags must be a list: {source}')

        phrases.append(
            PhraseConcept(
                name=name.strip(),
                surface_forms=tuple(str(form).strip() for form in surface_forms if str(form).strip()),
                definition=definition.strip(),
                tags=tuple(str(tag).strip() for tag in tags_raw if str(tag).strip()),
            ),
        )

    if not phrases:
        log.warning('glossary at %s contains no phrases', source)
    return phrases


def _surface_to_regex(surface: str) -> str:
    parts = [re.escape(part) for part in re.split(r'[ _-]+', surface.strip()) if part]
    if not parts:
        return re.escape(surface)
    if len(parts) == 1:
        return parts[0]
    return r'[ _-]+'.join(parts)


def _build_pattern_entries(phrases: Sequence[PhraseConcept]) -> tuple[_PatternEntry, ...]:
    entries: list[_PatternEntry] = []
    for phrase in phrases:
        for surface_form in phrase.surface_forms:
            entries.append(
                _PatternEntry(
                    regex=_surface_to_regex(surface_form),
                    phrase=phrase,
                    surface_form=surface_form,
                ),
            )
    entries.sort(key=lambda entry: len(entry.surface_form), reverse=True)
    return tuple(entries)


def _compile_match_regex(entries: Sequence[_PatternEntry]) -> re.Pattern[str]:
    if not entries:
        return re.compile(r'(?!x)x')

    grouped = '|'.join(f'({entry.regex})' for entry in entries)
    return re.compile(grouped, re.IGNORECASE)


def _dedupe_overlaps(hits: Sequence[PhraseHit]) -> list[PhraseHit]:
    if not hits:
        return []

    by_length = sorted(
        hits,
        key=lambda hit: (hit.span[1] - hit.span[0], -hit.span[0]),
        reverse=True,
    )
    kept: list[PhraseHit] = []
    for hit in by_length:
        if any(_spans_overlap(hit.span, existing.span) for existing in kept):
            continue
        kept.append(hit)
    return sorted(kept, key=lambda hit: hit.span[0])


def _spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]
