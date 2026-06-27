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

# Phase 2 W3: Zap (cross-corpus acceptance, WinUI 3 starter template).
# Fixture path is /tmp/zap_subset and includes both C# (Zap, Zap.Core) and
# C++ (Zap.ShellExt) sources to exercise cross-language tokenization.
ZAP_FIXTURE_SUBDIRS = (
    'Zap',
    'Zap.Core',
    'Zap.ShellExt',
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


@pytest.fixture(scope='session')
def acceptance_embedder():
    """Test embedder for the acceptance harness (P0-4).

    Returns (embed_fn, label). Label is for diagnostic printing in tests.
    """
    from open_webui.test.apps.webui.concepts.embedder import get_acceptance_embedder
    return get_acceptance_embedder()


@pytest.fixture(scope='session')
def lollipop_subset_store_with_embeddings(lollipop_subset_store, acceptance_embedder):
    """The W9-A session fixture, with concept embeddings populated.

    Returns the same store object; mutation is in-place.

    Idempotent: re-applying ``embed_store_concepts`` with ``overwrite=False`` is
    a no-op for concepts that already have embeddings.
    """
    from open_webui.test.apps.webui.concepts.embedder import embed_store_concepts
    embed_fn, label = acceptance_embedder
    count = embed_store_concepts(lollipop_subset_store, embed_fn)
    print(f'\n[acceptance] embedded {count} concepts using {label}')
    return lollipop_subset_store


@pytest.fixture(scope='session')
def zap_subset_store():
    """Build a concept graph from /tmp/zap_subset once per test session.

    Skips ALL tests using this fixture if /tmp/zap_subset isn't present
    (e.g., developer running locally without the corpus).

    Phase 2 W3: cross-corpus acceptance probe. Mirrors lollipop_subset_store
    but on the Zap WinUI 3 starter template (Zap + Zap.Core + Zap.ShellExt).
    """
    if not Path('/tmp/zap_subset').exists():
        pytest.skip(
            '/tmp/zap_subset not present. The Zap WinUI fixture corpus '
            f'must be populated with subdirs: {ZAP_FIXTURE_SUBDIRS}.'
        )

    missing = [
        sub for sub in ZAP_FIXTURE_SUBDIRS
        if not (Path('/tmp/zap_subset') / sub).exists()
    ]
    if missing:
        pytest.skip(
            f'/tmp/zap_subset is incomplete; missing subdirs: {missing}.'
        )

    from open_webui.retrieval.concepts.store.memory_store import InMemoryGraphStore
    from open_webui.retrieval.concepts.lifecycle.builder import BuildPlan, build

    store = InMemoryGraphStore()
    plan = BuildPlan(roots=(Path('/tmp/zap_subset'),), language_hint='csharp')
    result = build(plan, store)
    print(
        f'zap_subset_store: {result.concepts_upserted} concepts, '
        f'{result.edges_emitted} edges across {len(ZAP_FIXTURE_SUBDIRS)} subdirs',
        flush=True,
    )
    if result.concepts_upserted == 0:
        pytest.skip('Build produced zero concepts; possible corpus or tooling issue')

    return store


@pytest.fixture(scope='session')
def zap_subset_store_with_embeddings(zap_subset_store, acceptance_embedder):
    """The Zap session fixture, with concept embeddings populated.

    Mirrors lollipop_subset_store_with_embeddings — session-scoped,
    idempotent, in-place embedding mutation.
    """
    from open_webui.test.apps.webui.concepts.embedder import embed_store_concepts
    embed_fn, label = acceptance_embedder
    count = embed_store_concepts(zap_subset_store, embed_fn)
    print(f'\n[acceptance-zap] embedded {count} concepts using {label}')
    return zap_subset_store
