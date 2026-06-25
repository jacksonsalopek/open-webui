# Lollipop acceptance set

Fixed developer-question corpus used to gate Phase 1 of the concept knowledge
graph. See `docs/CONCEPT_GRAPH_PHASE1.md` gate 3 ("Acceptance set green") and
`docs/CONCEPT_GRAPH.md` § "V1 scope" for the hypothesis under test.

## Why this exists

Phase 1 must demonstrate that concept-anchored retrieval beats (or matches)
vector-only RAG on realistic codebase questions. That comparison only works
with a **frozen, hand-verified question set** — otherwise improvements cannot
be attributed to the graph vs. ad-hoc query tuning.

`lollipop_v1.yaml` is that set: 10 questions phrased like a senior engineer
typing into chat, each with greppable expected signals and file paths verified
against the Lollipop repo at `/Users/jacksonsalopek/dev/startups/ventana/lollipop/`.

The future harness (`test_acceptance_lollipop.py`, step 10) loads this file,
runs each question through hybrid retrieval and vector-only baseline, and
scores recall@10 plus optional blind human review.

## YAML schema

| Field | Meaning |
|---|---|
| `version` | Schema version (currently `1`). |
| `codebase` | Short slug for the indexed repo (`lollipop`). |
| `generated_at` | ISO date the set was authored or last fully re-verified. |
| `description` | Free-text summary of the set's purpose. |
| `questions[].id` | Stable slug (`q01_…`) for logging and diffs. |
| `questions[].intent` | Query class — see invariants below. |
| `questions[].difficulty` | `easy` / `medium` / `hard`. |
| `questions[].question` | Exact string fed to the retrieval router. |
| `questions[].expected_files` | Repo-relative paths; **at least one** must appear in top-K retrieval for a recall hit. |
| `questions[].expected_concepts` | High-centrality concepts that should appear in the query's concept neighborhood (used when graph retrieval is enabled). |
| `questions[].expected_answer_signal` | Tokens a correct generated answer should mention — class names, method names, or distinctive phrases that literally appear in `expected_files`. |
| `questions[].notes` | Rationale for human reviewers doing blind A/B comparison; not consumed by the harness. |

## Adding or modifying questions

1. **Verify before you ship.** Every path in `expected_files` must exist in
   the Lollipop repo. Every token in `expected_answer_signal` must be greppable
   inside those files (Read or `rg`, not memory).
2. **Keep the invariants** (see next section) unless you are intentionally
   authoring `lollipop_v2.yaml`.
3. **Prefer realistic phrasing** over symbol names in the question text.
   Bad: "find CopyResult in ToolbarViewModel". Good: "where do we copy the
   toolbar's result text to the clipboard?"
4. **Update `generated_at`** when you re-verify the set against a new Lollipop
   revision.
5. Run the sanity checks at the bottom of this file before opening a PR.

## Difficulty and intent invariants

`lollipop_v1.yaml` must contain exactly **10** questions with this spread:

| Intent | Count | Role |
|---|---:|---|
| `find_symbol` | 3 | Direct lookup — baseline RAG should handle most of these. |
| `where_used` | 2 | Reverse lookup — graph `references` edges should help. |
| `explain_region` | 2 | Multi-chunk composition — tests context packing. |
| `find_concept` | 2 | Terminology-only queries — **primary graph advantage** candidates. |
| `generate_code` | 1 | Assembly across interface + wiring + docs. |

| Difficulty | Count |
|---|---:|
| `easy` | 3 |
| `medium` | 4 |
| `hard` | 3 |

Future versions (`lollipop_v2.yaml`) may deviate once the hypothesis evolves
or Lollipop's architecture shifts materially.

## Acceptance threshold

From `docs/CONCEPT_GRAPH_PHASE1.md` gate 3:

Hybrid (concept-anchored) retrieval must win on **recall@10** **OR** on
**human-judged answer quality** using a 5-2-3 win/tie/loss minimum across
3 blind reviewers.

If the acceptance set fails, Phase 1 stops and the design is revisited before
layering docs, PRs, or people onto the graph.

## Versioning

- **`lollipop_v1.yaml`** — initial 10-question set for the current Lollipop
  toolbar + LLM + Predict architecture (2026-06).
- **`lollipop_v2.yaml`** — create when Lollipop changes materially (e.g.
  toolbar extension model rewrite) or when expanding coverage beyond 10
  questions. Keep v1 frozen for longitudinal comparison.

## Sanity checks

Verify all referenced files exist:

```bash
for f in $(yq '.questions[].expected_files[]' \
  open-webui/backend/open_webui/retrieval/concepts/acceptance/lollipop_v1.yaml); do
  test -f "/Users/jacksonsalopek/dev/startups/ventana/lollipop/$f" \
    || echo "MISSING: $f"
done
```

Smoke-load and validate shape:

```python
import yaml, pathlib
from collections import Counter

p = pathlib.Path("open-webui/backend/open_webui/retrieval/concepts/acceptance/lollipop_v1.yaml")
d = yaml.safe_load(p.read_text())
qs = d["questions"]
required = {"id", "intent", "difficulty", "question", "expected_files",
            "expected_concepts", "expected_answer_signal", "notes"}
assert len(qs) == 10
print("intent:", Counter(q["intent"] for q in qs))
print("difficulty:", Counter(q["difficulty"] for q in qs))
for q in qs:
    assert required.issubset(q), q["id"]
print("schema OK")
```
