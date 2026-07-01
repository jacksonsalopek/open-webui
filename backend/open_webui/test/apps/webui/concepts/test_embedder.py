"""Unit tests for the acceptance-harness embedder helpers."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from types import MappingProxyType

from open_webui.retrieval.concepts.schema import (
    Artifact,
    ArtifactKind,
    Concept,
    ConceptKind,
    DefinesProps,
    Edge,
    EdgeType,
    IsNamedInProps,
    ReferencesProps,
    edge_with_props,
)
from open_webui.retrieval.concepts.store.memory_store import InMemoryGraphStore
from open_webui.test.apps.webui.concepts.embedder import (
    FAKE_EMBED_DIM,
    _build_concept_embedding_text,
    embed_store_concepts,
    fake_embedder,
)

_TS = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _concept(name: str) -> Concept:
    return Concept(
        id=0,
        name=name,
        kind=ConceptKind.ATOMIC,
        first_seen_at=_TS,
        last_seen_at=_TS,
        centrality_score=None,
        embedding=None,
        definition=None,
        language_hint=None,
        original_tokens=(),
    )


def _phrase_concept(name: str, definition: str) -> Concept:
    return Concept(
        id=0,
        name=name,
        kind=ConceptKind.PHRASE,
        first_seen_at=_TS,
        last_seen_at=_TS,
        centrality_score=None,
        embedding=None,
        definition=definition,
        language_hint=None,
        original_tokens=(),
    )


def _co_occurs_edge(src_id: int, dst_id: int, *, weight: float = 1.0) -> Edge:
    return Edge(
        type=EdgeType.CO_OCCURS_WITH,
        src_id=src_id,
        dst_id=dst_id,
        properties=MappingProxyType({"weight": weight, "chunk_count": 1}),
    )


def _artifact(path: str = "/src/Foo.cs") -> Artifact:
    return Artifact(
        id=0,
        kind=ArtifactKind.CHUNK,
        path=path,
        chunk_index=0,
        language="csharp",
        byte_start=0,
        byte_end=100,
        last_modified_at=_TS,
    )


def _link_structural_neighbors(
    store: InMemoryGraphStore,
    anchor_id: int,
    neighbor_ids: list[int],
    *,
    path: str = "/src/Foo.cs",
) -> None:
    """Mirror production: anchor IS_NAMED_IN artifact; artifact DEFINES siblings.

    Also adds REFERENCES from anchor to each neighbor so radius=1
    ``store.neighborhood`` returns sibling concepts (artifact hops alone
    need radius=2; embedder intentionally uses radius=1).
    """
    artifact_id = store.upsert_artifact(_artifact(path))
    store.upsert_edge(
        edge_with_props(
            src_id=anchor_id,
            dst_id=artifact_id,
            props=IsNamedInProps(first_seen_at=_TS),
        ),
    )
    for neighbor_id in neighbor_ids:
        store.upsert_edge(
            edge_with_props(
                src_id=artifact_id,
                dst_id=neighbor_id,
                props=DefinesProps(count=1),
            ),
        )
        store.upsert_edge(
            edge_with_props(
                src_id=anchor_id,
                dst_id=neighbor_id,
                props=ReferencesProps(count=1),
            ),
        )


def _dot(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return sum(x * y for x, y in zip(a, b))


def test_fake_embedder_deterministic():
    embed = fake_embedder()
    a = embed("toolbar clipboard copy")
    b = embed("toolbar clipboard copy")
    assert a == b


def test_fake_embedder_different_tokens_different_vectors():
    embed = fake_embedder()
    a = embed("foo bar")
    b = embed("baz qux")
    assert a != b
    assert _dot(a, b) < 0.5


def test_fake_embedder_token_overlap_increases_similarity():
    embed = fake_embedder()
    ab = embed("foo bar")
    fb = embed("foo baz")
    xy = embed("xxx yyy")
    assert _dot(ab, fb) > _dot(ab, xy)


def test_fake_embedder_empty_text_zero_vector():
    embed = fake_embedder()
    for text in ("", "  ", "\t"):
        vec = embed(text)
        assert len(vec) == FAKE_EMBED_DIM
        assert all(v == 0.0 for v in vec)


def test_fake_embedder_l2_normalized():
    embed = fake_embedder()
    vec = embed("selection service dispatcher")
    norm = math.sqrt(sum(x * x for x in vec))
    assert norm == 0.0 or abs(norm - 1.0) < 1e-6


def test_embed_store_concepts_populates_embeddings():
    store = InMemoryGraphStore()
    for name in ("toolbar", "clipboard", "selection"):
        store.upsert_concept(_concept(name))

    count = embed_store_concepts(store, fake_embedder())
    assert count == 3

    scores = store.pagerank()
    for concept_id in scores:
        concept = store.get_concept(concept_id)
        assert concept is not None
        assert concept.embedding is not None
        assert len(concept.embedding) == FAKE_EMBED_DIM


def test_embed_store_concepts_idempotent():
    store = InMemoryGraphStore()
    for name in ("alpha", "beta"):
        store.upsert_concept(_concept(name))

    embed_fn = fake_embedder()
    assert embed_store_concepts(store, embed_fn) == 2
    assert embed_store_concepts(store, embed_fn) == 0


def test_embed_store_concepts_overwrite_true():
    store = InMemoryGraphStore()
    for name in ("one", "two", "three"):
        store.upsert_concept(_concept(name))

    embed_fn = fake_embedder()
    assert embed_store_concepts(store, embed_fn) == 3
    assert embed_store_concepts(store, embed_fn, overwrite=True) == 3


def _assert_embeddings_match_names(store: InMemoryGraphStore, embed_fn) -> None:
    for concept in store.list_concepts():
        stored = store.get_concept(concept.id)
        assert stored is not None
        assert stored.embedding == embed_fn(concept.name)


def test_embed_store_concepts_name_mode_default():
    store = InMemoryGraphStore()
    for name in ("alpha", "beta"):
        store.upsert_concept(_concept(name))

    embed_fn = fake_embedder()
    embed_store_concepts(store, embed_fn)
    _assert_embeddings_match_names(store, embed_fn)


def test_embed_store_concepts_name_mode_explicit():
    store = InMemoryGraphStore()
    for name in ("gamma", "delta"):
        store.upsert_concept(_concept(name))

    embed_fn = fake_embedder()
    embed_store_concepts(store, embed_fn, enrichment="name")
    _assert_embeddings_match_names(store, embed_fn)


def test_embed_store_concepts_name_plus_neighbors_includes_neighbor_names():
    store = InMemoryGraphStore()
    svg_id = store.upsert_concept(_concept("svg"))
    proc_id = store.upsert_concept(_concept("processor"))
    opt_id = store.upsert_concept(_concept("optimize"))
    xel_id = store.upsert_concept(_concept("xelement"))
    _link_structural_neighbors(store, svg_id, [proc_id, opt_id, xel_id])

    embed_fn = fake_embedder()
    embed_store_concepts(store, embed_fn, enrichment="name_plus_neighbors")

    svg = store.get_concept(svg_id)
    assert svg is not None
    expected_text = _build_concept_embedding_text(
        svg,
        store,
        enrichment="name_plus_neighbors",
    )
    assert svg.embedding == embed_fn(expected_text)
    for neighbor in ("processor", "optimize", "xelement"):
        assert neighbor in expected_text


def test_embed_store_concepts_empty_neighborhood_falls_back_to_name():
    store = InMemoryGraphStore()
    lone_id = store.upsert_concept(_concept("isolated"))

    embed_fn = fake_embedder()
    embed_store_concepts(store, embed_fn, enrichment="name_plus_neighbors")

    lone = store.get_concept(lone_id)
    assert lone is not None
    assert lone.embedding == embed_fn("isolated")


def test_embed_store_concepts_caps_at_10_neighbors():
    store = InMemoryGraphStore()
    anchor_id = store.upsert_concept(_concept("anchor"))
    neighbor_ids = [
        store.upsert_concept(_concept(f"neighbor_{i:02d}")) for i in range(15)
    ]
    _link_structural_neighbors(store, anchor_id, neighbor_ids)

    anchor = store.get_concept(anchor_id)
    assert anchor is not None
    text = _build_concept_embedding_text(
        anchor,
        store,
        enrichment="name_plus_neighbors",
    )
    neighbor_tokens = [t for t in text.split()[1:] if t.startswith("neighbor_")]
    assert len(neighbor_tokens) == 10


def test_embed_store_concepts_skips_self_name():
    anchor = _concept("foo")
    same_name_neighbor = Concept(
        id=99,
        name="foo",
        kind=ConceptKind.ATOMIC,
        first_seen_at=_TS,
        last_seen_at=_TS,
        centrality_score=None,
        embedding=None,
        definition=None,
        language_hint=None,
        original_tokens=(),
    )
    other_neighbor = Concept(
        id=100,
        name="bar",
        kind=ConceptKind.ATOMIC,
        first_seen_at=_TS,
        last_seen_at=_TS,
        centrality_score=None,
        embedding=None,
        definition=None,
        language_hint=None,
        original_tokens=(),
    )

    class _MockStore:
        def neighborhood(self, concept_id, **kwargs):  # noqa: ARG002
            return [same_name_neighbor, other_neighbor]

    text = _build_concept_embedding_text(
        anchor,
        _MockStore(),
        enrichment="name_plus_neighbors",
    )
    assert text == "foo bar"
    assert text.count("foo") == 1


def test_embed_store_concepts_phrase_prepends_definition():
    store = InMemoryGraphStore()
    phrase_id = store.upsert_concept(
        _phrase_concept("phrase_name", "A curated phrase concept."),
    )
    n1_id = store.upsert_concept(_concept("alpha"))
    n2_id = store.upsert_concept(_concept("beta"))
    _link_structural_neighbors(store, phrase_id, [n1_id, n2_id])

    phrase = store.get_concept(phrase_id)
    assert phrase is not None
    text = _build_concept_embedding_text(
        phrase,
        store,
        enrichment="name_plus_neighbors",
    )
    assert text.startswith("phrase_name: A curated phrase concept.")
    assert "alpha" in text
    assert "beta" in text


def test_embed_store_concepts_overwrite_with_rich_mode():
    store = InMemoryGraphStore()
    svg_id = store.upsert_concept(_concept("svg"))
    proc_id = store.upsert_concept(_concept("processor"))
    _link_structural_neighbors(store, svg_id, [proc_id])

    embed_fn = fake_embedder()
    embed_store_concepts(store, embed_fn, enrichment="name")
    svg_name_only = store.get_concept(svg_id)
    assert svg_name_only is not None
    name_embedding = svg_name_only.embedding

    embed_store_concepts(
        store,
        embed_fn,
        enrichment="name_plus_neighbors",
        overwrite=True,
    )
    svg_rich = store.get_concept(svg_id)
    assert svg_rich is not None
    assert svg_rich.embedding != name_embedding
    assert svg_rich.embedding == embed_fn("svg processor")


class _BrokenNeighborhoodStore:
    def neighborhood(self, *args, **kwargs):  # noqa: ARG002
        raise RuntimeError("neighborhood should not be called")


def test_build_concept_embedding_text_name_mode():
    text = _build_concept_embedding_text(
        _concept("foo"),
        _BrokenNeighborhoodStore(),
        enrichment="name",
    )
    assert text == "foo"


def test_build_concept_embedding_text_caps_at_400_chars():
    store = InMemoryGraphStore()
    anchor_id = store.upsert_concept(_concept("anchor"))
    for i in range(10):
        neighbor_id = store.upsert_concept(
            _concept(f"verylongneighborname{i:02d}{'x' * 40}"),
        )
        store.upsert_edge(
            edge_with_props(
                src_id=anchor_id,
                dst_id=neighbor_id,
                props=ReferencesProps(count=1),
            ),
        )

    anchor = store.get_concept(anchor_id)
    assert anchor is not None
    text = _build_concept_embedding_text(
        anchor,
        store,
        enrichment="name_plus_neighbors",
    )
    assert len(text) <= 400
    assert text.endswith("…")


def test_build_concept_embedding_text_includes_co_occurs_with_neighbors():
    store = InMemoryGraphStore()
    command_id = store.upsert_concept(_concept("command"))
    palette_id = store.upsert_concept(_concept("palette"))
    process_id = store.upsert_concept(_concept("process"))
    artifact_id = store.upsert_artifact(_artifact("/src/CommandPaletteService.cs"))
    store.upsert_edge(
        edge_with_props(
            src_id=command_id,
            dst_id=artifact_id,
            props=IsNamedInProps(first_seen_at=_TS),
        ),
    )
    store.upsert_edge(
        edge_with_props(
            src_id=artifact_id,
            dst_id=palette_id,
            props=DefinesProps(count=1),
        ),
    )
    store.upsert_edge(
        edge_with_props(
            src_id=command_id,
            dst_id=palette_id,
            props=ReferencesProps(count=1),
        ),
    )
    store.upsert_edge(_co_occurs_edge(command_id, process_id))

    command = store.get_concept(command_id)
    assert command is not None
    text = _build_concept_embedding_text(
        command,
        store,
        enrichment="name_plus_neighbors",
    )
    assert "palette" in text
    assert "process" in text


def _link_is_named_in_artifacts(
    store: InMemoryGraphStore,
    concept_id: int,
    paths: list[str],
) -> None:
    for path in paths:
        artifact_id = store.upsert_artifact(_artifact(path))
        store.upsert_edge(
            edge_with_props(
                src_id=concept_id,
                dst_id=artifact_id,
                props=IsNamedInProps(first_seen_at=_TS),
            ),
        )


def test_embed_store_concepts_name_plus_artifact_snippet_includes_basenames():
    store = InMemoryGraphStore()
    webp_id = store.upsert_concept(_concept("webp"))
    _link_is_named_in_artifacts(
        store,
        webp_id,
        ["/src/VipsImageProcessor.cs", "/src/ImageFormat.cs"],
    )

    embed_fn = fake_embedder()
    embed_store_concepts(store, embed_fn, enrichment="name_plus_artifact_snippet")

    webp = store.get_concept(webp_id)
    assert webp is not None
    expected_text = _build_concept_embedding_text(
        webp,
        store,
        enrichment="name_plus_artifact_snippet",
    )
    assert webp.embedding == embed_fn(expected_text)
    assert webp.embedding == embed_fn(
        "webp VipsImageProcessor.cs ImageFormat.cs",
    )


def test_embed_store_concepts_name_plus_artifact_snippet_falls_back_when_method_absent():
    inner = InMemoryGraphStore()
    concept_id = inner.upsert_concept(_concept("standalone"))

    class _MockStoreWithoutArtifacts:
        def list_concepts(self):
            concept = inner.get_concept(concept_id)
            return [concept] if concept is not None else []

        def set_concept_embedding(self, cid, embedding):
            inner.set_concept_embedding(cid, embedding)

    embed_fn = fake_embedder()
    store = _MockStoreWithoutArtifacts()
    embed_store_concepts(store, embed_fn, enrichment="name_plus_artifact_snippet")

    concept = inner.get_concept(concept_id)
    assert concept is not None
    assert concept.embedding == embed_fn("standalone")


def test_embed_store_concepts_name_plus_artifact_snippet_empty_artifacts_falls_back():
    store = InMemoryGraphStore()
    lone_id = store.upsert_concept(_concept("orphan"))

    embed_fn = fake_embedder()
    embed_store_concepts(store, embed_fn, enrichment="name_plus_artifact_snippet")

    lone = store.get_concept(lone_id)
    assert lone is not None
    assert lone.embedding == embed_fn("orphan")


def test_embed_store_concepts_name_plus_artifact_snippet_caps_at_3_artifacts():
    store = InMemoryGraphStore()
    anchor_id = store.upsert_concept(_concept("anchor"))
    paths = [f"/src/Artifact{i:02d}.cs" for i in range(5)]
    _link_is_named_in_artifacts(store, anchor_id, paths)

    anchor = store.get_concept(anchor_id)
    assert anchor is not None
    text = _build_concept_embedding_text(
        anchor,
        store,
        enrichment="name_plus_artifact_snippet",
    )
    basenames = [t for t in text.split()[1:] if t.endswith(".cs")]
    assert len(basenames) == 3


def test_embed_store_concepts_name_plus_artifact_snippet_phrase_prepends_definition():
    store = InMemoryGraphStore()
    phrase_id = store.upsert_concept(
        _phrase_concept("webp", "WebP image format support."),
    )
    _link_is_named_in_artifacts(store, phrase_id, ["/src/VipsImageProcessor.cs"])

    phrase = store.get_concept(phrase_id)
    assert phrase is not None
    text = _build_concept_embedding_text(
        phrase,
        store,
        enrichment="name_plus_artifact_snippet",
    )
    assert text.startswith("webp: WebP image format support.")
    assert "VipsImageProcessor.cs" in text
