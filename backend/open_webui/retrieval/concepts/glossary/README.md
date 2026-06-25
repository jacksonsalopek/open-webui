# Concept glossary

This directory holds **phrase concepts** — terms of art whose meaning is
*not* just the composition of their parts. The concept graph treats most
tokens as atomic (`toolbar`, `extension`, `model`); only curated phrases
become `Concept(kind=PHRASE)` nodes with definitions.

Compositional compounds stay atomic. For example, "toolbar extension" in
Lollipop is two atomic concepts linked by co-occurrence, not a glossary
entry. Reserve this file for genuine vocabulary: MVVM roles, concurrency
defects, resilience patterns, and other domain terms a new engineer must
learn explicitly.

## YAML schema

```yaml
version: 1
phrases:
  - name: race-condition          # canonical id (lowercase-with-dash)
    surface_forms:                # how the phrase appears in prose or code comments
      - race condition
    definition: |                 # 1–3 sentences; required for PHRASE concepts
      A defect where outcome depends on timing of concurrent events.
    tags: [concurrency]           # optional categorization
```

Required fields per entry: `name`, `surface_forms` (non-empty list),
`definition` (non-empty string). Optional: `tags`.

## Adding an entry

1. Confirm the term is a term of art, not a compositional compound (see
   `docs/CONCEPT_GRAPH.md` → "Concept granularity").
2. Pick a canonical `name` in lowercase-with-dash form.
3. List 2–4 `surface_forms` covering spacing variants (`view model`,
   `view-model`), common abbreviations (`GC`, `DI`), and casing seen in
   comments or docs.
4. Write a short, precise `definition` (1–3 sentences).
5. Add optional `tags` for filtering or documentation.
6. Run `pytest open_webui/test/apps/webui/concepts/test_glossary.py`.

## Layering per-project glossaries

`default.yaml` ships with the retrieval package and covers cross-project
software vocabulary (MVVM, .NET, concurrency, resilience). Project-specific
terms live in additional YAML files configured via
`CONCEPT_GRAPH_GLOSSARY_PATHS` (colon-separated paths).

Load order matters: later files **override** earlier entries with the same
`name`.

```python
from open_webui.retrieval.concepts.extraction.glossary import Glossary

glossary = Glossary.from_paths([
    Glossary.default(),                    # bundled seed
    "/data/kb/lollipop-glossary.yaml",   # project overlay
])
```

Or merge manually:

```python
glossary = Glossary.default().merge(Glossary.from_yaml("lollipop-glossary.yaml"))
```

## Matching rules (summary)

The `Glossary.match(text)` API finds phrase hits in free text:

- **Case-insensitive** — `Race Condition` matches `race-condition`.
- **Flexible separators** — `view model`, `view-model`, and `view_model`
  are equivalent when listed as surface forms.
- **Longest match wins** — overlapping spans keep the longer phrase
  (`race condition` beats a hypothetical shorter `race`).
- **Identifier boundaries** — matches must not sit inside identifier-shaped
  runs. The character immediately before and after the span must not be an
  ASCII letter, digit, or underscore. This prevents `ViewModelFactory` or
  `view_model_factory` from matching `view-model`.
- **Non-overlapping output** — returned hits do not overlap and are sorted
  by start offset ascending.

See `extraction/glossary.py` for the implementation used by the concept
extractor (step 5).
