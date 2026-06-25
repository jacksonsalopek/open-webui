"""Project-configurable identifier tokenization for concept extraction.

Atomic concepts are derived from identifier-shaped strings by splitting
PascalCase, camelCase, and snake_case into lowercased tokens. The
extractor (step 5) wraps these tokens in ``Concept(kind=ATOMIC)`` records;
this module performs pure tokenization only.

Tricky-case behaviour (defaults with ``keep_acronyms_intact=True``):

- **PascalCase acronyms:** ``LlmService`` → ``llm, service`` — consecutive
  uppercase letters form one acronym token until the last uppercase starts
  a mixed-case word (``HTMLDivElement`` → ``html, div, element``).
- **Mixed acronym boundaries:** ``GenAi`` splits on lower→upper transitions
  into ``gen, ai`` (not ``genai``), because ``Ai`` is uppercase + lowercase.
- **Interface prefix:** with ``strip_interface_prefix=True``, a leading ``I``
  followed by an uppercase letter is dropped entirely
  (``IObservableObject`` → ``observable, object``). With the flag off, ``i``
  is emitted but filtered by ``min_token_length=2``.
- **Generics:** with ``strip_generics=True``, angle-bracket content is
  discarded; ``Dictionary<string, int>`` → ``dictionary``.
- **Attributes:** brackets are stripped and parenthetical arguments removed;
  ``[RelayCommand(CanExecute = ...)]`` → ``relay, command``.
- **Numbers:** ``Http2Connection`` → ``http, connection`` when
  ``split_numbers=True`` and ``keep_pure_numeric_tokens=False`` (default).
  Pure-digit tokens (``13``, ``8080``) are dropped — magic numbers and port
  literals rarely encode semantic concepts. Alphanumeric tokens (``64KB``,
  ``utf8``, ``0xFF``) survive because they contain alphabetic characters.
- **Dunder names:** all leading/trailing underscores are stripped before
  snake_case splitting; ``__dunder_method__`` → ``dunder, method``.
- **Short PascalCase acronyms:** with
  ``emit_short_pascal_acronym_merges=True`` (default), adjacent PascalCase
  parts that are *both* short and purely alphabetic also emit the merged
  lowercase form as an additional token: ``NoOp`` → ``no, op, noop``;
  ``LooksLikeNoOp`` → ``looks, like, no, op, noop``. Solves the
  query-vs-source mismatch where the query says ``noop`` but the source's
  identifier splits to two separate tokens. The threshold is bounded by
  ``short_pascal_merge_max_part_len`` (default 2) so longer compounds like
  ``HttpModelDownloader`` are not collapsed; common acronym-style entries
  (``GenAi``, ``ViaUia``) where one part is 3 chars are handled separately
  via the router's query-side decomposition fallback rather than at
  tokenization time.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from typing import Literal

log = logging.getLogger(__name__)

# PascalCase / camelCase splitter: words, acronyms, digit runs.
_CAMEL_PART_RE = re.compile(
    r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)|\d+',
)

# Identifier-shaped substrings in free text (dotted names, generics, nullable).
_IDENTIFIER_IN_TEXT_RE = re.compile(
    r'\[[^\]]+\]'  # attribute markers
    r'|[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*'
    r'(?:<[^>]*>)?(?:\?)?',
)

# Parenthetical groups (attribute args, generic-adjacent noise in attributes).
_PARENS_RE = re.compile(r'\([^)]*\)')

# Generic type arguments.
_GENERICS_RE = re.compile(r'<[^>]*>')

# Attribute square brackets.
_ATTRIBUTE_BRACKETS_RE = re.compile(r'^\[(.+)\]$')

# Short English prepositions / fillers that, when they appear as adjacent
# PascalCase parts (e.g. ``ById``, ``UpTo``, ``IsOk``), should NOT trigger an
# acronym-style merge. Kept local to this module so identifier tokenization
# remains free of cross-package imports; the canonical stopword set in
# ``stopwords.py`` is consulted later by the extractor / router.
_PASCAL_MERGE_BLOCKLIST: frozenset[str] = frozenset(
    {
        'as', 'at', 'be', 'by', 'do', 'go', 'if', 'in', 'is', 'it', 'me',
        'my', 'of', 'on', 'or', 'so', 'to', 'up', 'us', 'we',
    },
)


@dataclass(frozen=True, slots=True)
class TokenRules:
    """Tunable tokenization rules for a project or language."""

    strip_interface_prefix: bool = True
    strip_underscore_prefix: bool = True
    strip_generics: bool = True
    strip_attribute_markers: bool = True
    strip_nullable_marker: bool = True
    keep_acronyms_intact: bool = True
    split_numbers: bool = True
    keep_pure_numeric_tokens: bool = False
    lowercase: bool = True
    min_token_length: int = 2
    dotted_name_handling: Literal['split', 'keep'] = 'split'
    emit_short_pascal_acronym_merges: bool = True
    short_pascal_merge_max_part_len: int = 2

    def _replace(self, **kwargs: object) -> TokenRules:
        """Return a copy with selected fields overridden (test / config helper)."""
        return replace(self, **kwargs)


CSHARP_DEFAULT_RULES = TokenRules(
    strip_interface_prefix=True,
    strip_underscore_prefix=True,
    strip_generics=True,
    strip_attribute_markers=True,
    strip_nullable_marker=True,
    keep_acronyms_intact=True,
    split_numbers=True,
    lowercase=True,
    min_token_length=2,
    dotted_name_handling='split',
)

PYTHON_DEFAULT_RULES = TokenRules(
    strip_interface_prefix=False,
    strip_underscore_prefix=True,
    strip_generics=False,
    strip_attribute_markers=False,
    strip_nullable_marker=False,
    keep_acronyms_intact=True,
    split_numbers=True,
    lowercase=True,
    min_token_length=2,
    dotted_name_handling='split',
)

TYPESCRIPT_DEFAULT_RULES = TokenRules(
    strip_interface_prefix=True,
    strip_underscore_prefix=False,
    strip_generics=True,
    strip_attribute_markers=False,
    strip_nullable_marker=True,
    keep_acronyms_intact=True,
    split_numbers=True,
    lowercase=True,
    min_token_length=2,
    dotted_name_handling='split',
)

_LANGUAGE_PRESETS: dict[str, TokenRules] = {
    'csharp': CSHARP_DEFAULT_RULES,
    'c_sharp': CSHARP_DEFAULT_RULES,
    'python': PYTHON_DEFAULT_RULES,
    'typescript': TYPESCRIPT_DEFAULT_RULES,
    'tsx': TYPESCRIPT_DEFAULT_RULES,
    'javascript': TYPESCRIPT_DEFAULT_RULES,
}


def rules_for_language(language: str) -> TokenRules:
    """Return the default ``TokenRules`` preset for a tree-sitter language name."""
    return _LANGUAGE_PRESETS.get(language.lower(), CSHARP_DEFAULT_RULES)


def tokenize(identifier: str, *, rules: TokenRules) -> list[str]:
    """Tokenize a single identifier into a lowercased token sequence."""
    if not identifier or not identifier.strip():
        return []

    text = identifier.strip()
    tokens: list[str] = []

    for segment in _split_dotted(text, rules):
        tokens.extend(_tokenize_segment(segment, rules))

    return _finalize_tokens(tokens, rules)


def tokenize_text(text: str, *, rules: TokenRules) -> list[str]:
    """Extract identifier-shaped substrings from free text and tokenize each.

    Pure-numeric tokens (no alphabetic characters) are dropped by default
    (``keep_pure_numeric_tokens=False``). Magic numbers, port literals, and
    version fragments dominate real corpora but carry little semantic signal;
    meaningful numeric standards are expected to appear as glossary phrases.
    """
    if not text or not text.strip():
        return []

    tokens: list[str] = []
    for match in _IDENTIFIER_IN_TEXT_RE.finditer(text):
        tokens.extend(tokenize(match.group(0), rules=rules))
    return tokens


def _split_dotted(text: str, rules: TokenRules) -> list[str]:
    if rules.dotted_name_handling == 'keep' or '.' not in text:
        return [text]
    return [part for part in text.split('.') if part]


def _tokenize_segment(segment: str, rules: TokenRules) -> list[str]:
    work = segment

    if rules.strip_attribute_markers:
        bracket_match = _ATTRIBUTE_BRACKETS_RE.match(work)
        if bracket_match:
            work = bracket_match.group(1)
            work = _PARENS_RE.sub('', work)

    if rules.strip_generics:
        work = _GENERICS_RE.sub('', work)

    if rules.strip_nullable_marker and work.endswith('?'):
        work = work[:-1]

    work = work.strip()
    if not work:
        return []

    if rules.strip_underscore_prefix:
        work = work.lstrip('_')
    work = work.rstrip('_')
    if not work:
        return []

    if rules.strip_interface_prefix and _should_strip_interface_prefix(work):
        work = work[1:]

    parts = work.split('_') if '_' in work else [work]
    tokens: list[str] = []
    for part in parts:
        if not part:
            continue
        tokens.extend(_split_identifier_part(part, rules))
    return tokens


def _should_strip_interface_prefix(name: str) -> bool:
    return len(name) > 1 and name[0] == 'I' and name[1].isupper()


def _split_identifier_part(part: str, rules: TokenRules) -> list[str]:
    if rules.split_numbers and rules.keep_acronyms_intact:
        parts = _CAMEL_PART_RE.findall(part)
    elif not rules.keep_acronyms_intact:
        # Naive split on every upper/lower and digit boundary.
        parts = re.findall(r'[A-Za-z]+|\d+', part)
    elif not rules.split_numbers:
        # Letters only — merge digit runs into adjacent words via regex tweak.
        parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)', part)
    else:
        parts = _CAMEL_PART_RE.findall(part)

    if rules.emit_short_pascal_acronym_merges and len(parts) >= 2:
        merges = _short_pascal_merges(parts, rules.short_pascal_merge_max_part_len)
        if merges:
            parts = parts + merges
    return parts


def _short_pascal_merges(parts: list[str], max_part_len: int) -> list[str]:
    """Emit merged acronym-style tokens for adjacent short alphabetic parts.

    Solves the ``NoOp`` mismatch: source identifier ``LooksLikeNoOp`` splits
    to ``looks, like, no, op`` while a query token ``noop`` is a single
    lowercase chunk. Without this merge, the query never resolves to the
    source's atomic concepts. Restricted to adjacent parts whose lengths are
    both ≤ ``max_part_len`` (default 2) so longer compounds like
    ``HttpModelDownloader`` are not collapsed into noise.
    """
    if max_part_len < 2:
        return []
    merges: list[str] = []
    for i in range(len(parts) - 1):
        left, right = parts[i], parts[i + 1]
        if not left.isalpha() or not right.isalpha():
            continue
        # Require both parts to be at least 2 chars; single-char parts get
        # dropped by ``min_token_length`` anyway, so merging across them
        # would imply an adjacency that isn't observable in the final token
        # stream (and produces noise tokens like ``xff`` from ``0xFF``).
        if len(left) < 2 or len(right) < 2:
            continue
        if len(left) > max_part_len or len(right) > max_part_len:
            continue
        if (
            left.lower() in _PASCAL_MERGE_BLOCKLIST
            or right.lower() in _PASCAL_MERGE_BLOCKLIST
        ):
            continue
        merges.append(left + right)
    return merges


def _has_alpha_char(token: str) -> bool:
    return any(character.isalpha() for character in token)


def _finalize_tokens(raw: list[str], rules: TokenRules) -> list[str]:
    result: list[str] = []
    for token in raw:
        value = token.lower() if rules.lowercase else token
        if not rules.keep_pure_numeric_tokens and not _has_alpha_char(value):
            continue
        if value.isdigit() or len(value) >= rules.min_token_length:
            result.append(value)
    return result
