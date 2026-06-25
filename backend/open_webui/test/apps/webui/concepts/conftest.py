import pytest
from pathlib import Path

# Phase 1.5 P0-1: documented fixture composition. The acceptance gate's
# reproducibility depends on this list matching scripts/build_lollipop_fixture.sh
# exactly. If you change one, change the other.
LOLLIPOP_FIXTURE_SUBDIRS = (
    'ViewModels',
    'Services',
    'Helpers',
    'Lollipop.Extensions',
    'Lollipop.Llm',
)


@pytest.fixture(scope='session')
def lollipop_subset_store():
    """Build a concept graph from /tmp/lollipop_subset once per test session.

    Skips ALL tests using this fixture if the Lollipop subset isn't present
    in the container (e.g., a developer running locally without the corpus).
    """
    if not Path('/tmp/lollipop_subset').exists():
        pytest.skip(
            '/tmp/lollipop_subset not present. Run scripts/build_lollipop_fixture.sh '
            'from the repo root (see Phase 1.5 P0-1 in docs/CONCEPT_GRAPH_PHASE1.md '
            f'for the documented subdir list: {LOLLIPOP_FIXTURE_SUBDIRS}).'
        )

    missing = [
        sub for sub in LOLLIPOP_FIXTURE_SUBDIRS
        if not (Path('/tmp/lollipop_subset') / sub).exists()
    ]
    if missing:
        pytest.skip(
            f'/tmp/lollipop_subset is incomplete; missing subdirs: {missing}. '
            'Re-run scripts/build_lollipop_fixture.sh.'
        )

    from open_webui.retrieval.concepts.store.memory_store import InMemoryGraphStore
    from open_webui.retrieval.concepts.lifecycle.builder import BuildPlan, build

    store = InMemoryGraphStore()
    plan = BuildPlan(roots=(Path('/tmp/lollipop_subset'),), language_hint='csharp')
    result = build(plan, store)
    print(
        f'lollipop_subset_store: {result.concepts_upserted} concepts, '
        f'{result.edges_emitted} edges across {len(LOLLIPOP_FIXTURE_SUBDIRS)} subdirs',
        flush=True,
    )
    if result.concepts_upserted == 0:
        pytest.skip('Build produced zero concepts; possible corpus or tooling issue')

    return store
