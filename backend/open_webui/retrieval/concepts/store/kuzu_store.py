"""KuzuDB-backed ``GraphStore`` for Phase 1 concept graph persistence.

Schema layout: **Option A** — one relationship table per ``EdgeType`` with
typed property columns matching the ``*Props`` dataclasses in ``schema.py``.
This follows Kuzu's preferred typed-rel shape, keeps per-type indexes simple,
and mirrors the ERD in ``CONCEPT_GRAPH.md``.  The trade-off is that
``edge_types=None`` neighborhood / path queries over multiple rel kinds require
UNION or Python-side adjacency assembly; we load normalized adjacency in Python
for BFS / PageRank so behavior matches ``InMemoryGraphStore`` exactly
(including the contract-test pattern of using concept ids as artifact endpoints
for ``DEFINES`` — production ingest in step 7 will write real ``Artifact``
nodes).

``concept_name_kind`` uniqueness: Kuzu 0.11.x has no composite unique secondary
index, so idempotent upsert enforces ``(name, kind)`` in Python (with a one-
time warning logged at schema init).
"""

from __future__ import annotations

import logging
import math
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any

import kuzu

from open_webui.retrieval.concepts.schema import (
    Artifact,
    ArtifactKind,
    Concept,
    ConceptKind,
    Edge,
    EdgeType,
    concept_from_dict,
)
from open_webui.retrieval.concepts.store.memory_store import (
    _PAGERANK_TOLERANCE,
    _artifact_idempotency_key,
    merge_edge_properties,
)
from open_webui.retrieval.concepts.store.protocol import EdgeFilter, GraphTransaction

log = logging.getLogger(__name__)

# Edge merge / property column metadata per rel table (Option A).
_EDGE_REL: dict[EdgeType, dict[str, Any]] = {
    EdgeType.DEFINES: {
        'table': 'Defines',
        'from_label': 'Artifact',
        'to_label': 'Concept',
        'columns': ('count',),
    },
    EdgeType.REFERENCES: {
        'table': 'References',
        'from_label': 'Artifact',
        'to_label': 'Concept',
        'columns': ('count', 'positions'),
    },
    EdgeType.CO_OCCURS_WITH: {
        'table': 'CoOccursWith',
        'from_label': 'Concept',
        'to_label': 'Concept',
        'columns': ('weight', 'chunk_count'),
    },
    EdgeType.IS_NAMED_IN: {
        'table': 'IsNamedIn',
        'from_label': 'Concept',
        'to_label': 'Artifact',
        'columns': ('first_seen_at',),
    },
    EdgeType.IS_CANONICAL_ALIAS_OF: {
        'table': 'IsCanonicalAliasOf',
        'from_label': 'Concept',
        'to_label': 'Concept',
        'columns': ('introduced_at', 'planned_removal_at', 'removed_at'),
    },
}

_ALIAS_CHAIN_LIMIT = 100
_VECTOR_INDEX_NAME = 'concept_emb_idx'


def _parse_dt(value: object) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _row_value(row: Any, index: int = 0) -> Any:
    if hasattr(row, '__getitem__'):
        return row[index]
    return row


class _KuzuTransaction:
    """Maps ``GraphTransaction`` to Kuzu ``BEGIN`` / ``COMMIT`` / ``ROLLBACK``."""

    def __init__(self, conn: kuzu.Connection) -> None:
        self._conn = conn
        self._committed = False

    def commit(self) -> None:
        self._conn.execute('COMMIT')
        self._committed = True

    def rollback(self) -> None:
        self._conn.execute('ROLLBACK')
        self._committed = True

    def __enter__(self) -> _KuzuTransaction:
        self._conn.execute('BEGIN TRANSACTION')
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        if exc_type is not None and not self._committed:
            self.rollback()


class KuzuGraphStore:
    """Embedded Kuzu ``GraphStore`` — all Cypher lives in this module."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        embedding_dim: int = 2560,
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 200,
    ) -> None:
        self._db_path = str(db_path)
        self._embedding_dim = embedding_dim
        self._hnsw_m = hnsw_m
        self._hnsw_ef_construction = hnsw_ef_construction
        self._db = kuzu.Database(self._db_path)
        self._conn = kuzu.Connection(self._db)
        self._vector_index_ready = False
        if self._is_fresh_db():
            self._create_schema()
            self._create_vector_index()
        else:
            self._vector_index_ready = self._vector_index_exists()
        self._next_concept_id = self._load_max_concept_id() + 1
        self._next_artifact_id = self._load_max_artifact_id() + 1

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _is_fresh_db(self) -> bool:
        try:
            result = self._conn.execute(
                "CALL SHOW_TABLES() RETURN name",
            )
            names = {_row_value(row, 0) for row in result}
            return 'Concept' not in names
        except Exception:
            return True

    def _try_execute(self, query: str, params: dict[str, Any] | None = None) -> None:
        try:
            if params:
                self._conn.execute(query, params)
            else:
                self._conn.execute(query)
        except RuntimeError as exc:
            msg = str(exc).lower()
            if 'already exist' in msg:
                return
            raise

    def _create_schema(self) -> None:
        dim = self._embedding_dim
        self._try_execute(
            f"""
            CREATE NODE TABLE Concept(
                id INT64,
                name STRING,
                kind STRING,
                first_seen_at TIMESTAMP,
                last_seen_at TIMESTAMP,
                centrality_score DOUBLE,
                embedding FLOAT[{dim}],
                definition STRING,
                language_hint STRING,
                original_tokens STRING[],
                PRIMARY KEY(id)
            )
            """,
        )
        self._try_execute(
            """
            CREATE NODE TABLE Artifact(
                id INT64,
                kind STRING,
                path STRING,
                chunk_index INT64,
                language STRING,
                byte_start INT64,
                byte_end INT64,
                last_modified_at TIMESTAMP,
                PRIMARY KEY(id)
            )
            """,
        )
        self._try_execute(
            'CREATE REL TABLE Defines(FROM Artifact TO Concept, count INT64)',
        )
        self._try_execute(
            'CREATE REL TABLE References(FROM Artifact TO Concept, count INT64, positions INT64[])',
        )
        self._try_execute(
            'CREATE REL TABLE CoOccursWith(FROM Concept TO Concept, weight DOUBLE, chunk_count INT64)',
        )
        self._try_execute(
            'CREATE REL TABLE IsNamedIn(FROM Concept TO Artifact, first_seen_at TIMESTAMP)',
        )
        self._try_execute(
            """
            CREATE REL TABLE IsCanonicalAliasOf(
                FROM Concept TO Concept,
                introduced_at TIMESTAMP,
                planned_removal_at TIMESTAMP,
                removed_at TIMESTAMP
            )
            """,
        )
        log.warning(
            'Kuzu 0.11.x has no composite unique index on Concept(name, kind); '
            'uniqueness is enforced in upsert_concept()',
        )

    def _vector_index_exists(self) -> bool:
        try:
            result = self._conn.execute(
                f"CALL SHOW_INDEXES() RETURN name WHERE name = '{_VECTOR_INDEX_NAME}'",
            )
            return any(True for _ in result)
        except Exception:
            return False

    def _pad_embedding(self, embedding: Sequence[float] | None) -> list[float] | None:
        if embedding is None:
            return None
        vec = [float(x) for x in embedding]
        if len(vec) >= self._embedding_dim:
            return vec[: self._embedding_dim]
        return vec + [0.0] * (self._embedding_dim - len(vec))

    def _create_vector_index(self) -> None:
        # Kuzu 0.11.3 accepts metric but not m / ef_construction; hnsw_m and
        # hnsw_ef_construction are kept on the constructor for Neo4j parity.
        try:
            self._conn.execute(
                f"""
                CALL CREATE_VECTOR_INDEX(
                    'Concept',
                    '{_VECTOR_INDEX_NAME}',
                    'embedding',
                    metric := 'cosine'
                )
                """,
            )
            self._vector_index_ready = True
        except Exception as exc:
            log.warning('vector index creation failed (%s); brute-force fallback active', exc)
            self._vector_index_ready = False

    def _load_max_concept_id(self) -> int:
        result = self._conn.execute('MATCH (c:Concept) RETURN max(c.id)')
        for row in result:
            val = _row_value(row, 0)
            if val is None:
                return 0
            return int(val)
        return 0

    def _load_max_artifact_id(self) -> int:
        result = self._conn.execute('MATCH (a:Artifact) RETURN max(a.id)')
        for row in result:
            val = _row_value(row, 0)
            if val is None:
                return 0
            return int(val)
        return 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_edge_types(self, edge_types: EdgeFilter) -> set[EdgeType]:
        if edge_types is None:
            return {EdgeType.CO_OCCURS_WITH}
        return set(edge_types)

    def _artifact_row_to_model(self, row: Any) -> Artifact:
        (
            aid,
            kind,
            path,
            chunk_index,
            language,
            byte_start,
            byte_end,
            last_modified_at,
        ) = row
        return Artifact(
            id=int(aid),
            kind=ArtifactKind(str(kind)),
            path=str(path),
            chunk_index=int(chunk_index) if chunk_index is not None else None,
            language=str(language) if language is not None else None,
            byte_start=int(byte_start) if byte_start is not None else None,
            byte_end=int(byte_end) if byte_end is not None else None,
            last_modified_at=_parse_dt(last_modified_at),
        )

    def _lookup_artifact_id_and_kind(
        self,
        artifact: Artifact,
    ) -> tuple[int, ArtifactKind] | None:
        if artifact.kind == ArtifactKind.CHUNK:
            result = self._conn.execute(
                """
                MATCH (a:Artifact {path: $path, chunk_index: $chunk_index})
                RETURN a.id, a.kind
                LIMIT 1
                """,
                {'path': artifact.path, 'chunk_index': artifact.chunk_index},
            )
        else:
            result = self._conn.execute(
                """
                MATCH (a:Artifact {path: $path})
                WHERE a.chunk_index IS NULL
                RETURN a.id, a.kind
                LIMIT 1
                """,
                {'path': artifact.path},
            )
        for row in result:
            return int(_row_value(row, 0)), ArtifactKind(str(_row_value(row, 1)))
        return None

    def _merge_tokens(self, existing: Sequence[str], incoming: Sequence[str]) -> list[str]:
        return list(dict.fromkeys((*existing, *incoming)))

    def _concept_row_to_dict(self, row: Any) -> dict[str, Any]:
        (
            cid,
            name,
            kind,
            first_seen_at,
            last_seen_at,
            centrality_score,
            embedding,
            definition,
            language_hint,
            original_tokens,
        ) = row
        emb: list[float] | None = None
        if embedding is not None:
            emb = [float(x) for x in embedding]
        tokens: list[str] = []
        if original_tokens is not None:
            tokens = [str(x) for x in original_tokens]
        return {
            'id': int(cid),
            'name': str(name),
            'kind': str(kind),
            'first_seen_at': _parse_dt(first_seen_at).isoformat(),
            'last_seen_at': _parse_dt(last_seen_at).isoformat(),
            'centrality_score': (
                float(centrality_score) if centrality_score is not None else None
            ),
            'embedding': emb,
            'definition': str(definition) if definition is not None else None,
            'language_hint': str(language_hint) if language_hint is not None else None,
            'original_tokens': tokens,
        }

    def _fetch_edge_properties(
        self,
        edge_type: EdgeType,
        src_id: int,
        dst_id: int,
    ) -> dict[str, object] | None:
        meta = _EDGE_REL[edge_type]
        table = meta['table']
        from_label = meta['from_label']
        to_label = meta['to_label']
        cols = meta['columns']
        col_list = ', '.join(f'r.{c}' for c in cols)
        query = f"""
            MATCH (a:{from_label} {{id: $src_id}})-[r:{table}]->(b:{to_label} {{id: $dst_id}})
            RETURN {col_list}
        """
        result = self._conn.execute(query, {'src_id': src_id, 'dst_id': dst_id})
        for row in result:
            props: dict[str, object] = {}
            for i, col in enumerate(cols):
                val = _row_value(row, i)
                if val is None:
                    continue
                if col.endswith('_at'):
                    props[col] = _parse_dt(val).isoformat()
                elif col == 'positions':
                    props[col] = [int(x) for x in val]
                elif col == 'count' or col == 'chunk_count':
                    props[col] = int(val)
                elif col == 'weight':
                    props[col] = float(val)
                else:
                    props[col] = val
            return props
        return None

    def _set_edge_properties(
        self,
        edge_type: EdgeType,
        src_id: int,
        dst_id: int,
        props: Mapping[str, object],
    ) -> None:
        meta = _EDGE_REL[edge_type]
        table = meta['table']
        from_label = meta['from_label']
        to_label = meta['to_label']
        set_clause = ', '.join(f'r.{c} = ${c}' for c in meta['columns'] if c in props)
        params: dict[str, Any] = {'src_id': src_id, 'dst_id': dst_id}
        for key, val in props.items():
            if key.endswith('_at') and isinstance(val, str):
                params[key] = _parse_dt(val)
            else:
                params[key] = val
        query = f"""
            MATCH (a:{from_label} {{id: $src_id}})-[r:{table}]->(b:{to_label} {{id: $dst_id}})
            SET {set_clause}
        """
        self._conn.execute(query, params)

    def _create_edge(
        self,
        edge_type: EdgeType,
        src_id: int,
        dst_id: int,
        props: Mapping[str, object],
    ) -> None:
        meta = _EDGE_REL[edge_type]
        table = meta['table']
        from_label = meta['from_label']
        to_label = meta['to_label']
        cols = meta['columns']
        params: dict[str, Any] = {'src_id': src_id, 'dst_id': dst_id}
        for key in cols:
            val = props[key]
            if key.endswith('_at') and isinstance(val, str):
                params[key] = _parse_dt(val)
            else:
                params[key] = val
        props_clause = ', '.join(f'{c}: ${c}' for c in cols)
        query = f"""
            MATCH (a:{from_label} {{id: $src_id}}), (b:{to_label} {{id: $dst_id}})
            CREATE (a)-[:{table} {{{props_clause}}}]->(b)
        """
        self._conn.execute(query, params)

    def _load_out_neighbors(
        self,
        allowed: set[EdgeType],
    ) -> dict[int, list[tuple[int, float]]]:
        """Directed adjacency normalized to concept-id space with edge weights."""
        adj: dict[int, list[tuple[int, float]]] = {}
        if EdgeType.CO_OCCURS_WITH in allowed:
            result = self._conn.execute(
                """
                MATCH (a:Concept)-[r:CoOccursWith]->(b:Concept)
                RETURN a.id, b.id, r.weight
                ORDER BY a.id, r.weight DESC, b.id ASC
                """,
            )
            for row in result:
                src, dst = int(_row_value(row, 0)), int(_row_value(row, 1))
                weight = float(_row_value(row, 2))
                adj.setdefault(src, []).append((dst, weight))
        if EdgeType.IS_CANONICAL_ALIAS_OF in allowed:
            result = self._conn.execute(
                """
                MATCH (a:Concept)-[:IsCanonicalAliasOf]->(b:Concept)
                RETURN a.id, b.id
                ORDER BY a.id, b.id ASC
                """,
            )
            for row in result:
                src, dst = int(_row_value(row, 0)), int(_row_value(row, 1))
                adj.setdefault(src, []).append((dst, 1.0))
        if EdgeType.DEFINES in allowed:
            result = self._conn.execute(
                """
                MATCH (a:Artifact)-[:Defines]->(b:Concept)
                RETURN a.id, b.id
                ORDER BY a.id, b.id ASC
                """,
            )
            for row in result:
                src, dst = int(_row_value(row, 0)), int(_row_value(row, 1))
                adj.setdefault(src, []).append((dst, 1.0))
        if EdgeType.REFERENCES in allowed:
            result = self._conn.execute(
                """
                MATCH (a:Artifact)-[:References]->(b:Concept)
                RETURN a.id, b.id
                ORDER BY a.id, b.id ASC
                """,
            )
            for row in result:
                src, dst = int(_row_value(row, 0)), int(_row_value(row, 1))
                adj.setdefault(src, []).append((dst, 1.0))
        if EdgeType.IS_NAMED_IN in allowed:
            result = self._conn.execute(
                """
                MATCH (a:Concept)-[:IsNamedIn]->(b:Artifact)
                RETURN a.id, b.id
                ORDER BY a.id, b.id ASC
                """,
            )
            for row in result:
                src, dst = int(_row_value(row, 0)), int(_row_value(row, 1))
                adj.setdefault(src, []).append((dst, 1.0))
        return adj

    def _all_concept_ids(self) -> list[int]:
        result = self._conn.execute('MATCH (c:Concept) RETURN c.id ORDER BY c.id')
        return [int(_row_value(row, 0)) for row in result]

    def _batch_create_concepts(self, rows: Sequence[Mapping[str, Any]]) -> None:
        # Kuzu 0.11 UNWIND-struct batch CREATE with all-null DOUBLE columns fails
        # (STRUCT_EXTRACT(row, centrality_score) inferred as STRING). Insert required
        # columns first, then SET optional columns in follow-up passes.
        create_rows = [
            {
                'id': r['id'],
                'name': r['name'],
                'kind': r['kind'],
                'first_seen_at': r['first_seen_at'],
                'last_seen_at': r['last_seen_at'],
                'original_tokens': r['original_tokens'],
            }
            for r in rows
        ]
        self._conn.execute(
            """
            UNWIND $rows AS row
            CREATE (c:Concept {
                id: row.id,
                name: row.name,
                kind: row.kind,
                first_seen_at: row.first_seen_at,
                last_seen_at: row.last_seen_at,
                original_tokens: row.original_tokens
            })
            """,
            {'rows': create_rows},
        )

        embedding_rows = [
            {'id': r['id'], 'embedding': r['embedding']}
            for r in rows
            if r.get('embedding') is not None
        ]
        if embedding_rows:
            self._conn.execute(
                """
                UNWIND $rows AS row
                MATCH (c:Concept {id: row.id})
                SET c.embedding = row.embedding
                """,
                {'rows': embedding_rows},
            )

        centrality_rows = [
            {'id': r['id'], 'centrality_score': r['centrality_score']}
            for r in rows
            if r.get('centrality_score') is not None
        ]
        if centrality_rows:
            self._conn.execute(
                """
                UNWIND $rows AS row
                MATCH (c:Concept {id: row.id})
                SET c.centrality_score = row.centrality_score
                """,
                {'rows': centrality_rows},
            )

        definition_rows = [
            {'id': r['id'], 'definition': r['definition']}
            for r in rows
            if r.get('definition') is not None
        ]
        if definition_rows:
            self._conn.execute(
                """
                UNWIND $rows AS row
                MATCH (c:Concept {id: row.id})
                SET c.definition = row.definition
                """,
                {'rows': definition_rows},
            )

        language_rows = [
            {'id': r['id'], 'language_hint': r['language_hint']}
            for r in rows
            if r.get('language_hint') is not None
        ]
        if language_rows:
            self._conn.execute(
                """
                UNWIND $rows AS row
                MATCH (c:Concept {id: row.id})
                SET c.language_hint = row.language_hint
                """,
                {'rows': language_rows},
            )

    def _batch_create_artifacts(self, rows: Sequence[Mapping[str, Any]]) -> None:
        # Same Kuzu 0.11 NULL typing constraint as concepts: all-null INT64 columns
        # in a UNWIND struct batch fail binding. Insert required columns first.
        create_rows = [
            {
                'id': r['id'],
                'kind': r['kind'],
                'path': r['path'],
                'chunk_index': r['chunk_index'],
                'last_modified_at': r['last_modified_at'],
            }
            for r in rows
        ]
        self._conn.execute(
            """
            UNWIND $rows AS row
            CREATE (a:Artifact {
                id: row.id,
                kind: row.kind,
                path: row.path,
                chunk_index: row.chunk_index,
                last_modified_at: row.last_modified_at
            })
            """,
            {'rows': create_rows},
        )

        language_rows = [
            {'id': r['id'], 'language': r['language']}
            for r in rows
            if r.get('language') is not None
        ]
        if language_rows:
            self._conn.execute(
                """
                UNWIND $rows AS row
                MATCH (a:Artifact {id: row.id})
                SET a.language = row.language
                """,
                {'rows': language_rows},
            )

        byte_rows = [
            {
                'id': r['id'],
                'byte_start': r['byte_start'],
                'byte_end': r['byte_end'],
            }
            for r in rows
            if r.get('byte_start') is not None or r.get('byte_end') is not None
        ]
        if byte_rows:
            self._conn.execute(
                """
                UNWIND $rows AS row
                MATCH (a:Artifact {id: row.id})
                SET a.byte_start = row.byte_start,
                    a.byte_end = row.byte_end
                """,
                {'rows': byte_rows},
            )

    # ------------------------------------------------------------------
    # GraphStore API
    # ------------------------------------------------------------------

    def upsert_concept(self, concept: Concept) -> int:
        result = self._conn.execute(
            """
            MATCH (c:Concept {name: $name, kind: $kind})
            RETURN c.id, c.original_tokens
            """,
            {'name': concept.name, 'kind': concept.kind.value},
        )
        for row in result:
            existing_id = int(_row_value(row, 0))
            existing_tokens = _row_value(row, 1) or []
            merged = self._merge_tokens(existing_tokens, concept.original_tokens)
            self._conn.execute(
                """
                MATCH (c:Concept {id: $id})
                SET c.last_seen_at = $last_seen_at,
                    c.original_tokens = $original_tokens
                """,
                {
                    'id': existing_id,
                    'last_seen_at': concept.last_seen_at,
                    'original_tokens': merged,
                },
            )
            return existing_id

        new_id = self._next_concept_id
        self._next_concept_id += 1
        emb = self._pad_embedding(concept.embedding)
        self._conn.execute(
            """
            CREATE (c:Concept {
                id: $id,
                name: $name,
                kind: $kind,
                first_seen_at: $first_seen_at,
                last_seen_at: $last_seen_at,
                centrality_score: $centrality_score,
                embedding: $embedding,
                definition: $definition,
                language_hint: $language_hint,
                original_tokens: $original_tokens
            })
            """,
            {
                'id': new_id,
                'name': concept.name,
                'kind': concept.kind.value,
                'first_seen_at': concept.first_seen_at,
                'last_seen_at': concept.last_seen_at,
                'centrality_score': concept.centrality_score,
                'embedding': emb,
                'definition': concept.definition,
                'language_hint': concept.language_hint,
                'original_tokens': list(concept.original_tokens),
            },
        )
        return new_id

    def _collect_incident_edges(self, concept_id: int) -> list[Edge]:
        """Return every edge touching ``concept_id`` (both directions)."""
        edges: list[Edge] = []
        seen: set[tuple[EdgeType, int, int]] = set()

        for edge_type, meta in _EDGE_REL.items():
            table = meta['table']
            from_label = meta['from_label']
            to_label = meta['to_label']

            if from_label == 'Concept':
                result = self._conn.execute(
                    f"""
                    MATCH (a:Concept {{id: $cid}})-[r:{table}]->(b:{to_label})
                    RETURN b.id
                    """,
                    {'cid': concept_id},
                )
                for row in result:
                    dst_id = int(_row_value(row, 0))
                    key = (edge_type, concept_id, dst_id)
                    if key in seen:
                        continue
                    props = self._fetch_edge_properties(edge_type, concept_id, dst_id)
                    if props is None:
                        continue
                    seen.add(key)
                    edges.append(
                        Edge(
                            type=edge_type,
                            src_id=concept_id,
                            dst_id=dst_id,
                            properties=MappingProxyType(dict(props)),
                        ),
                    )

            if to_label == 'Concept' and from_label != 'Concept':
                result = self._conn.execute(
                    f"""
                    MATCH (a:{from_label})-[r:{table}]->(b:Concept {{id: $cid}})
                    RETURN a.id
                    """,
                    {'cid': concept_id},
                )
                for row in result:
                    src_id = int(_row_value(row, 0))
                    key = (edge_type, src_id, concept_id)
                    if key in seen:
                        continue
                    props = self._fetch_edge_properties(edge_type, src_id, concept_id)
                    if props is None:
                        continue
                    seen.add(key)
                    edges.append(
                        Edge(
                            type=edge_type,
                            src_id=src_id,
                            dst_id=concept_id,
                            properties=MappingProxyType(dict(props)),
                        ),
                    )
            elif to_label == 'Concept' and from_label == 'Concept':
                result = self._conn.execute(
                    f"""
                    MATCH (a:Concept)-[r:{table}]->(b:Concept {{id: $cid}})
                    RETURN a.id
                    """,
                    {'cid': concept_id},
                )
                for row in result:
                    src_id = int(_row_value(row, 0))
                    key = (edge_type, src_id, concept_id)
                    if key in seen:
                        continue
                    props = self._fetch_edge_properties(edge_type, src_id, concept_id)
                    if props is None:
                        continue
                    seen.add(key)
                    edges.append(
                        Edge(
                            type=edge_type,
                            src_id=src_id,
                            dst_id=concept_id,
                            properties=MappingProxyType(dict(props)),
                        ),
                    )

        return edges

    def _reinsert_concept_with_embedding(
        self,
        concept: Concept,
        embedding_payload: list[float] | None,
    ) -> None:
        """Replace a concept row when Kuzu blocks SET on vector-indexed embeddings."""
        concept_id = concept.id
        incident = self._collect_incident_edges(concept_id)
        self._conn.execute(
            'MATCH (c:Concept {id: $id}) DETACH DELETE c',
            {'id': concept_id},
        )
        self._conn.execute(
            """
            CREATE (c:Concept {
                id: $id,
                name: $name,
                kind: $kind,
                first_seen_at: $first_seen_at,
                last_seen_at: $last_seen_at,
                centrality_score: $centrality_score,
                embedding: $embedding,
                definition: $definition,
                language_hint: $language_hint,
                original_tokens: $original_tokens
            })
            """,
            {
                'id': concept_id,
                'name': concept.name,
                'kind': concept.kind.value,
                'first_seen_at': concept.first_seen_at,
                'last_seen_at': concept.last_seen_at,
                'centrality_score': concept.centrality_score,
                'embedding': embedding_payload,
                'definition': concept.definition,
                'language_hint': concept.language_hint,
                'original_tokens': list(concept.original_tokens),
            },
        )
        for edge in incident:
            self._create_edge(
                edge.type,
                edge.src_id,
                edge.dst_id,
                edge.properties,
            )

    def set_concept_embedding(
        self,
        concept_id: int,
        embedding: tuple[float, ...] | None,
    ) -> None:
        if embedding == ():
            raise ValueError(
                f'set_concept_embedding({concept_id}): empty-tuple embedding is '
                f'rejected; pass None to clear instead.',
            )
        existing = self.get_concept(concept_id)
        if existing is None:
            raise KeyError(concept_id)

        embedding_payload = self._pad_embedding(embedding)
        try:
            self._conn.execute(
                'MATCH (c:Concept {id: $id}) SET c.embedding = $embedding',
                {'id': concept_id, 'embedding': embedding_payload},
            )
        except RuntimeError as exc:
            msg = str(exc)
            if (
                'used in one or more indexes' not in msg
                and 'does not exist in catalog' not in msg
            ):
                raise
            self._reinsert_concept_with_embedding(existing, embedding_payload)

    def upsert_concepts_batch(self, concepts: Sequence[Concept]) -> list[int]:
        if not concepts:
            return []

        unique_keys = list(dict.fromkeys((c.name, c.kind) for c in concepts))
        lookup_keys = [{'name': name, 'kind': kind.value} for name, kind in unique_keys]

        db_existing: dict[tuple[str, str], tuple[int, list[str]]] = {}
        if lookup_keys:
            result = self._conn.execute(
                """
                UNWIND $keys AS key
                MATCH (c:Concept {name: key.name, kind: key.kind})
                RETURN c.name, c.kind, c.id, c.original_tokens
                """,
                {'keys': lookup_keys},
            )
            for row in result:
                name = str(_row_value(row, 0))
                kind = str(_row_value(row, 1))
                cid = int(_row_value(row, 2))
                tokens_raw = _row_value(row, 3) or []
                db_existing[(name, kind)] = (
                    cid,
                    [str(x) for x in tokens_raw],
                )

        id_by_key: dict[tuple[str, ConceptKind], int] = {}
        merge_state: dict[tuple[str, ConceptKind], dict[str, Any]] = {}

        for concept in concepts:
            key = (concept.name, concept.kind)
            db_key = (concept.name, concept.kind.value)

            if key not in id_by_key:
                if db_key in db_existing:
                    existing_id, existing_tokens = db_existing[db_key]
                    id_by_key[key] = existing_id
                    merge_state[key] = {
                        'from_db': True,
                        'first': concept,
                        'tokens': list(existing_tokens),
                        'last_seen_at': concept.last_seen_at,
                    }
                else:
                    new_id = self._next_concept_id
                    self._next_concept_id += 1
                    id_by_key[key] = new_id
                    merge_state[key] = {
                        'from_db': False,
                        'first': concept,
                        'tokens': [],
                        'last_seen_at': concept.last_seen_at,
                    }

            state = merge_state[key]
            state['tokens'] = self._merge_tokens(state['tokens'], concept.original_tokens)
            state['last_seen_at'] = concept.last_seen_at

        insert_rows: list[dict[str, Any]] = []
        update_rows: list[dict[str, Any]] = []
        for key, state in merge_state.items():
            merged_tokens = state['tokens']
            last_seen = state['last_seen_at']
            if state['from_db']:
                update_rows.append(
                    {
                        'id': id_by_key[key],
                        'last_seen_at': last_seen,
                        'original_tokens': merged_tokens,
                    },
                )
            else:
                first: Concept = state['first']
                insert_rows.append(
                    {
                        'id': id_by_key[key],
                        'name': first.name,
                        'kind': first.kind.value,
                        'first_seen_at': first.first_seen_at,
                        'last_seen_at': last_seen,
                        'centrality_score': first.centrality_score,
                        'embedding': self._pad_embedding(first.embedding),
                        'definition': first.definition,
                        'language_hint': first.language_hint,
                        'original_tokens': merged_tokens,
                    },
                )

        if insert_rows:
            self._batch_create_concepts(insert_rows)

        if update_rows:
            # Kuzu 0.11 supports UNWIND + MATCH + SET; no MERGE-on-UNWIND needed.
            self._conn.execute(
                """
                UNWIND $rows AS row
                MATCH (c:Concept {id: row.id})
                SET c.last_seen_at = row.last_seen_at,
                    c.original_tokens = row.original_tokens
                """,
                {'rows': update_rows},
            )

        return [id_by_key[(c.name, c.kind)] for c in concepts]

    def upsert_artifact(self, artifact: Artifact) -> int:
        key = _artifact_idempotency_key(artifact)
        existing = self._lookup_artifact_id_and_kind(artifact)
        if existing is not None:
            existing_id, existing_kind = existing
            if existing_kind != artifact.kind:
                raise ValueError(
                    f'artifact kind mismatch on idempotency key {key!r}: '
                    f'existing={existing_kind!r}, incoming={artifact.kind!r}',
                )
            if artifact.kind == ArtifactKind.CHUNK:
                self._conn.execute(
                    """
                    MATCH (a:Artifact {path: $path, chunk_index: $chunk_index})
                    SET a.last_modified_at = $last_modified_at,
                        a.byte_start = $byte_start,
                        a.byte_end = $byte_end,
                        a.language = $language
                    """,
                    {
                        'path': artifact.path,
                        'chunk_index': artifact.chunk_index,
                        'last_modified_at': artifact.last_modified_at,
                        'byte_start': artifact.byte_start,
                        'byte_end': artifact.byte_end,
                        'language': artifact.language,
                    },
                )
            else:
                self._conn.execute(
                    """
                    MATCH (a:Artifact {path: $path})
                    WHERE a.chunk_index IS NULL
                    SET a.last_modified_at = $last_modified_at,
                        a.byte_start = $byte_start,
                        a.byte_end = $byte_end,
                        a.language = $language
                    """,
                    {
                        'path': artifact.path,
                        'last_modified_at': artifact.last_modified_at,
                        'byte_start': artifact.byte_start,
                        'byte_end': artifact.byte_end,
                        'language': artifact.language,
                    },
                )
            return existing_id

        new_id = self._next_artifact_id
        self._next_artifact_id += 1
        chunk_index = (
            artifact.chunk_index if artifact.kind == ArtifactKind.CHUNK else None
        )
        self._conn.execute(
            """
            CREATE (a:Artifact {
                id: $id,
                kind: $kind,
                path: $path,
                chunk_index: $chunk_index,
                language: $language,
                byte_start: $byte_start,
                byte_end: $byte_end,
                last_modified_at: $last_modified_at
            })
            """,
            {
                'id': new_id,
                'kind': artifact.kind.value,
                'path': artifact.path,
                'chunk_index': chunk_index,
                'language': artifact.language,
                'byte_start': artifact.byte_start,
                'byte_end': artifact.byte_end,
                'last_modified_at': artifact.last_modified_at,
            },
        )
        return new_id

    def upsert_artifacts_batch(self, artifacts: Sequence[Artifact]) -> list[int]:
        if not artifacts:
            return []

        chunks = [a for a in artifacts if a.kind == ArtifactKind.CHUNK]
        files = [a for a in artifacts if a.kind != ArtifactKind.CHUNK]

        db_existing: dict[tuple[str, int | None], tuple[int, ArtifactKind]] = {}

        if chunks:
            chunk_keys = [
                {'path': a.path, 'chunk_index': a.chunk_index} for a in chunks
            ]
            unique_chunk_keys = list(
                dict.fromkeys((k['path'], k['chunk_index']) for k in chunk_keys),
            )
            lookup = [
                {'path': path, 'chunk_index': idx} for path, idx in unique_chunk_keys
            ]
            result = self._conn.execute(
                """
                UNWIND $keys AS key
                MATCH (a:Artifact {path: key.path, chunk_index: key.chunk_index})
                RETURN a.path, a.chunk_index, a.id, a.kind
                """,
                {'keys': lookup},
            )
            for row in result:
                path = str(_row_value(row, 0))
                chunk_index = _row_value(row, 1)
                cid = int(_row_value(row, 2))
                kind = ArtifactKind(str(_row_value(row, 3)))
                db_existing[(path, int(chunk_index) if chunk_index is not None else None)] = (
                    cid,
                    kind,
                )

        if files:
            file_paths = list(dict.fromkeys(a.path for a in files))
            result = self._conn.execute(
                """
                UNWIND $paths AS path
                MATCH (a:Artifact {path: path})
                WHERE a.chunk_index IS NULL
                RETURN a.path, a.id, a.kind
                """,
                {'paths': file_paths},
            )
            for row in result:
                path = str(_row_value(row, 0))
                cid = int(_row_value(row, 1))
                kind = ArtifactKind(str(_row_value(row, 2)))
                db_existing[(path, None)] = (cid, kind)

        id_by_key: dict[tuple[str, int | None], int] = {}
        pending_insert: dict[tuple[str, int | None], dict[str, Any]] = {}
        pending_update: dict[int, dict[str, Any]] = {}

        for artifact in artifacts:
            key = _artifact_idempotency_key(artifact)
            mutable = {
                'last_modified_at': artifact.last_modified_at,
                'byte_start': artifact.byte_start,
                'byte_end': artifact.byte_end,
                'language': artifact.language,
            }

            if key in id_by_key:
                if key in pending_insert:
                    pending_insert[key].update(mutable)
                else:
                    pending_update[id_by_key[key]] = {
                        'id': id_by_key[key],
                        **mutable,
                    }
                continue

            if key in db_existing:
                existing_id, existing_kind = db_existing[key]
                if existing_kind != artifact.kind:
                    raise ValueError(
                        f'artifact kind mismatch on idempotency key {key!r}: '
                        f'existing={existing_kind!r}, incoming={artifact.kind!r}',
                    )
                id_by_key[key] = existing_id
                pending_update[existing_id] = {'id': existing_id, **mutable}
            else:
                new_id = self._next_artifact_id
                self._next_artifact_id += 1
                id_by_key[key] = new_id
                chunk_index = (
                    artifact.chunk_index if artifact.kind == ArtifactKind.CHUNK else None
                )
                pending_insert[key] = {
                    'id': new_id,
                    'kind': artifact.kind.value,
                    'path': artifact.path,
                    'chunk_index': chunk_index,
                    **mutable,
                }

        insert_rows = list(pending_insert.values())
        update_rows = list(pending_update.values())

        if insert_rows:
            self._batch_create_artifacts(insert_rows)

        if update_rows:
            self._conn.execute(
                """
                UNWIND $rows AS row
                MATCH (a:Artifact {id: row.id})
                SET a.last_modified_at = row.last_modified_at,
                    a.byte_start = row.byte_start,
                    a.byte_end = row.byte_end,
                    a.language = row.language
                """,
                {'rows': update_rows},
            )

        return [id_by_key[_artifact_idempotency_key(a)] for a in artifacts]

    def upsert_edge(self, edge: Edge) -> None:
        existing = self._fetch_edge_properties(edge.type, edge.src_id, edge.dst_id)
        if existing is not None:
            merged = merge_edge_properties(edge.type, existing, edge.properties)
            self._set_edge_properties(edge.type, edge.src_id, edge.dst_id, merged)
            return

        props = dict(edge.properties)
        for key, val in props.items():
            if isinstance(val, datetime):
                props[key] = val.isoformat()
        self._create_edge(edge.type, edge.src_id, edge.dst_id, props)

    def upsert_edges_batch(self, edges: Sequence[Edge]) -> None:
        if not edges:
            return

        by_type: dict[EdgeType, list[Edge]] = {}
        for edge in edges:
            by_type.setdefault(edge.type, []).append(edge)

        for edge_type, group in by_type.items():
            self._upsert_edges_batch_for_type(edge_type, group)

    def _upsert_edges_batch_for_type(
        self,
        edge_type: EdgeType,
        edges: Sequence[Edge],
    ) -> None:
        meta = _EDGE_REL[edge_type]
        table = meta['table']
        from_label = meta['from_label']
        to_label = meta['to_label']
        cols = meta['columns']

        self._validate_edge_endpoints_batch(edge_type, edges)

        pairs = [{'src_id': e.src_id, 'dst_id': e.dst_id} for e in edges]
        existing_props: dict[tuple[int, int], dict[str, object]] = {}
        result = self._conn.execute(
            f"""
            UNWIND $pairs AS pair
            MATCH (a:{from_label} {{id: pair.src_id}})-[r:{table}]->(b:{to_label} {{id: pair.dst_id}})
            RETURN pair.src_id, pair.dst_id, {', '.join(f'r.{c}' for c in cols)}
            """,
            {'pairs': pairs},
        )
        for row in result:
            src_id = int(_row_value(row, 0))
            dst_id = int(_row_value(row, 1))
            props: dict[str, object] = {}
            for i, col in enumerate(cols):
                val = _row_value(row, i + 2)
                if val is None:
                    continue
                if col.endswith('_at'):
                    props[col] = _parse_dt(val).isoformat()
                elif col == 'positions':
                    props[col] = [int(x) for x in val]
                elif col in ('count', 'chunk_count'):
                    props[col] = int(val)
                elif col == 'weight':
                    props[col] = float(val)
                else:
                    props[col] = val
            existing_props[(src_id, dst_id)] = props

        create_rows: list[dict[str, Any]] = []
        pending_create: dict[tuple[int, int], dict[str, Any]] = {}
        pending_update: dict[tuple[int, int], dict[str, Any]] = {}

        def _row_from_props(src_id: int, dst_id: int, props: Mapping[str, object]) -> dict[str, Any]:
            row: dict[str, Any] = {'src_id': src_id, 'dst_id': dst_id}
            for col in cols:
                if col not in props:
                    continue
                val = props[col]
                if val is None:
                    continue
                row[col] = (
                    _parse_dt(val) if col.endswith('_at') and isinstance(val, str) else val
                )
            return row

        for edge in edges:
            pair_key = (edge.src_id, edge.dst_id)
            incoming = dict(edge.properties)
            for key, val in incoming.items():
                if isinstance(val, datetime):
                    incoming[key] = val.isoformat()

            if pair_key in existing_props:
                merged = merge_edge_properties(
                    edge_type,
                    existing_props[pair_key],
                    incoming,
                )
                existing_props[pair_key] = merged
                row = _row_from_props(edge.src_id, edge.dst_id, merged)
                if pair_key in pending_create:
                    pending_create[pair_key] = row
                else:
                    pending_update[pair_key] = row
            else:
                existing_props[pair_key] = incoming
                pending_create[pair_key] = _row_from_props(
                    edge.src_id,
                    edge.dst_id,
                    incoming,
                )

        create_rows = list(pending_create.values())
        update_rows = list(pending_update.values())

        if edge_type == EdgeType.REFERENCES:
            self._batch_create_references_edges(create_rows)
            self._batch_update_references_edges(update_rows)
            return

        if create_rows:
            props_clause = ', '.join(f'{c}: row.{c}' for c in cols)
            self._conn.execute(
                f"""
                UNWIND $rows AS row
                MATCH (a:{from_label} {{id: row.src_id}}), (b:{to_label} {{id: row.dst_id}})
                CREATE (a)-[:{table} {{{props_clause}}}]->(b)
                """,
                {'rows': create_rows},
            )

        if update_rows:
            set_clause = ', '.join(f'r.{c} = row.{c}' for c in cols)
            self._conn.execute(
                f"""
                UNWIND $rows AS row
                MATCH (a:{from_label} {{id: row.src_id}})-[r:{table}]->(b:{to_label} {{id: row.dst_id}})
                SET {set_clause}
                """,
                {'rows': update_rows},
            )

    def _batch_create_references_edges(
        self,
        rows: Sequence[Mapping[str, Any]],
    ) -> None:
        if not rows:
            return
        # Homogeneous UNWIND structs: omit optional ``positions`` from the CREATE
        # batch (mixed Some/None breaks Kuzu 0.11 struct-array binding).
        count_rows = [
            {'src_id': r['src_id'], 'dst_id': r['dst_id'], 'count': r['count']}
            for r in rows
        ]
        self._conn.execute(
            """
            UNWIND $rows AS row
            MATCH (a:Artifact {id: row.src_id}), (b:Concept {id: row.dst_id})
            CREATE (a)-[:References {count: row.count}]->(b)
            """,
            {'rows': count_rows},
        )
        position_rows = [
            {
                'src_id': r['src_id'],
                'dst_id': r['dst_id'],
                'positions': [int(x) for x in r['positions']],
            }
            for r in rows
            if r.get('positions')
        ]
        if position_rows:
            self._conn.execute(
                """
                UNWIND $rows AS row
                MATCH (a:Artifact {id: row.src_id})-[r:References]->(b:Concept {id: row.dst_id})
                SET r.positions = row.positions
                """,
                {'rows': position_rows},
            )

    def _batch_update_references_edges(
        self,
        rows: Sequence[Mapping[str, Any]],
    ) -> None:
        if not rows:
            return
        count_rows = [
            {'src_id': r['src_id'], 'dst_id': r['dst_id'], 'count': r['count']}
            for r in rows
        ]
        self._conn.execute(
            """
            UNWIND $rows AS row
            MATCH (a:Artifact {id: row.src_id})-[r:References]->(b:Concept {id: row.dst_id})
            SET r.count = row.count
            """,
            {'rows': count_rows},
        )
        position_rows = [
            {
                'src_id': r['src_id'],
                'dst_id': r['dst_id'],
                'positions': [int(x) for x in r['positions']],
            }
            for r in rows
            if r.get('positions')
        ]
        if position_rows:
            self._conn.execute(
                """
                UNWIND $rows AS row
                MATCH (a:Artifact {id: row.src_id})-[r:References]->(b:Concept {id: row.dst_id})
                SET r.positions = row.positions
                """,
                {'rows': position_rows},
            )

    def _validate_edge_endpoints_batch(
        self,
        edge_type: EdgeType,
        edges: Sequence[Edge],
    ) -> None:
        meta = _EDGE_REL[edge_type]
        from_label = meta['from_label']
        to_label = meta['to_label']

        src_ids = {e.src_id for e in edges}
        dst_ids = {e.dst_id for e in edges}

        if from_label == 'Concept':
            self._assert_nodes_exist('Concept', src_ids, 'src_id')
        else:
            self._assert_nodes_exist('Artifact', src_ids, 'src_id')

        if to_label == 'Concept':
            self._assert_nodes_exist('Concept', dst_ids, 'dst_id')
        else:
            self._assert_nodes_exist('Artifact', dst_ids, 'dst_id')

    def _assert_nodes_exist(
        self,
        label: str,
        node_ids: set[int],
        role: str,
    ) -> None:
        if not node_ids:
            return
        result = self._conn.execute(
            f"""
            UNWIND $ids AS id
            MATCH (n:{label} {{id: id}})
            RETURN id
            """,
            {'ids': list(node_ids)},
        )
        found = {int(_row_value(row, 0)) for row in result}
        missing = node_ids - found
        if missing:
            missing_id = next(iter(missing))
            raise ValueError(f'missing {label.lower()} endpoint {role}={missing_id}')

    def get_concept(self, concept_id: int) -> Concept | None:
        result = self._conn.execute(
            """
            MATCH (c:Concept {id: $id})
            RETURN c.id, c.name, c.kind, c.first_seen_at, c.last_seen_at,
                   c.centrality_score, c.embedding, c.definition,
                   c.language_hint, c.original_tokens
            """,
            {'id': concept_id},
        )
        for row in result:
            return concept_from_dict(self._concept_row_to_dict(row))
        return None

    def list_concepts(self) -> Iterable[Concept]:
        result = self._conn.execute(
            'MATCH (c:Concept) RETURN c.id ORDER BY c.id ASC',
        )
        ids: list[int] = []
        while result.has_next():
            row = result.get_next()
            ids.append(int(row[0]))
        for cid in ids:
            concept = self.get_concept(cid)
            if concept is not None:
                yield concept

    def find_concept_by_name(
        self,
        name: str,
        kind: ConceptKind | None = None,
    ) -> int | None:
        if kind is not None:
            result = self._conn.execute(
                """
                MATCH (c:Concept)
                WHERE c.name = $name AND c.kind = $kind
                RETURN c.id
                LIMIT 1
                """,
                {'name': name, 'kind': kind.value},
            )
        else:
            result = self._conn.execute(
                """
                MATCH (c:Concept)
                WHERE c.name = $name
                RETURN c.id
                LIMIT 1
                """,
                {'name': name},
            )
        for row in result:
            return int(_row_value(row, 0))
        return None

    def get_artifact(self, artifact_id: int) -> Artifact | None:
        result = self._conn.execute(
            """
            MATCH (a:Artifact {id: $id})
            RETURN a.id, a.kind, a.path, a.chunk_index, a.language,
                   a.byte_start, a.byte_end, a.last_modified_at
            """,
            {'id': artifact_id},
        )
        for row in result:
            return self._artifact_row_to_model(row)
        return None

    def _artifact_edge_weight_from_row(
        self,
        edge_type: EdgeType,
        weight_val: object,
    ) -> float:
        if weight_val is not None:
            return float(weight_val)
        return 1.0

    def list_artifacts_for_concept(
        self,
        concept_id: int,
        *,
        edge_types: EdgeFilter = (EdgeType.IS_NAMED_IN,),
        limit: int | None = None,
    ) -> list[Artifact]:
        if self.get_concept(concept_id) is None:
            raise KeyError(concept_id)

        allowed = set(edge_types) if edge_types is not None else {EdgeType.IS_NAMED_IN}
        artifact_weights: dict[int, float] = {}

        for edge_type in allowed:
            if edge_type == EdgeType.CO_OCCURS_WITH:
                continue
            meta = _EDGE_REL[edge_type]
            table = meta['table']
            from_label = meta['from_label']
            to_label = meta['to_label']

            if from_label == 'Concept' and to_label == 'Artifact':
                weight_expr = '1.0'
                query = f"""
                    MATCH (c:Concept {{id: $concept_id}})-[r:{table}]->(a:Artifact)
                    RETURN a.id, {weight_expr} AS weight
                """
            elif from_label == 'Artifact' and to_label == 'Concept':
                weight_col = 'r.count' if 'count' in meta['columns'] else '1.0'
                query = f"""
                    MATCH (a:Artifact)-[r:{table}]->(c:Concept {{id: $concept_id}})
                    RETURN a.id, {weight_col} AS weight
                """
            else:
                continue

            result = self._conn.execute(query, {'concept_id': concept_id})
            for row in result:
                artifact_id = int(_row_value(row, 0))
                weight = self._artifact_edge_weight_from_row(
                    edge_type,
                    _row_value(row, 1),
                )
                existing = artifact_weights.get(artifact_id)
                if existing is None or weight > existing:
                    artifact_weights[artifact_id] = weight

        ranked = sorted(artifact_weights.items(), key=lambda item: (-item[1], item[0]))
        if limit is not None:
            ranked = ranked[:limit]

        artifacts: list[Artifact] = []
        for artifact_id, _weight in ranked:
            artifact = self.get_artifact(artifact_id)
            if artifact is not None:
                artifacts.append(artifact)
        return artifacts

    def neighborhood(
        self,
        anchor_id: int,
        *,
        radius: int = 1,
        edge_types: EdgeFilter = None,
        limit: int = 100,
    ) -> list[Concept]:
        resolved = self.resolve_alias(anchor_id)
        allowed = self._normalize_edge_types(edge_types)
        if self.get_concept(resolved) is None or radius < 1 or limit < 1:
            return []

        adj = self._load_out_neighbors(allowed)
        seen: set[int] = {resolved}
        frontier: list[int] = [resolved]
        collected: list[Concept] = []

        for _ in range(radius):
            next_frontier: list[int] = []
            for node_id in frontier:
                for neighbor_id, _weight in adj.get(node_id, []):
                    if neighbor_id in seen:
                        continue
                    seen.add(neighbor_id)
                    next_frontier.append(neighbor_id)
                    concept = self.get_concept(neighbor_id)
                    if concept is not None:
                        collected.append(concept)
                        if len(collected) >= limit:
                            return collected[:limit]
            frontier = next_frontier
            if not frontier:
                break

        return collected[:limit]

    def shortest_path(
        self,
        src_id: int,
        dst_id: int,
        *,
        edge_types: EdgeFilter = None,
        max_hops: int = 6,
    ) -> list[Concept]:
        if src_id == dst_id:
            concept = self.get_concept(src_id)
            return [concept] if concept is not None else []

        allowed = self._normalize_edge_types(edge_types)
        if self.get_concept(src_id) is None or self.get_concept(dst_id) is None:
            return []

        adj = self._load_out_neighbors(allowed)
        parent: dict[int, int | None] = {src_id: None}
        queue: deque[tuple[int, int]] = deque([(src_id, 0)])
        found = False

        while queue:
            current, depth = queue.popleft()
            if depth >= max_hops:
                continue
            for neighbor_id, _weight in adj.get(current, []):
                if neighbor_id in parent:
                    continue
                parent[neighbor_id] = current
                if neighbor_id == dst_id:
                    found = True
                    queue.clear()
                    break
                queue.append((neighbor_id, depth + 1))

        if not found:
            return []

        path_ids: list[int] = []
        cursor: int | None = dst_id
        while cursor is not None:
            path_ids.append(cursor)
            cursor = parent[cursor]
        path_ids.reverse()
        result: list[Concept] = []
        for cid in path_ids:
            concept = self.get_concept(cid)
            if concept is not None:
                result.append(concept)
        return result

    def pagerank(
        self,
        *,
        edge_types: EdgeFilter = None,
        damping: float = 0.85,
        iterations: int = 20,
    ) -> dict[int, float]:
        allowed = self._normalize_edge_types(edge_types)
        nodes = self._all_concept_ids()
        if not nodes:
            return {}

        index = {node_id: i for i, node_id in enumerate(nodes)}
        n = len(nodes)
        out_neighbors: list[list[int]] = [[] for _ in range(n)]
        adj = self._load_out_neighbors(allowed)

        for src_id, neighbors in adj.items():
            if src_id not in index:
                continue
            src_i = index[src_id]
            for dst_id, _weight in neighbors:
                if dst_id not in index:
                    continue
                out_neighbors[src_i].append(index[dst_id])

        scores = [1.0 / n] * n
        teleport = (1.0 - damping) / n

        for _ in range(iterations):
            new_scores = [teleport] * n
            dangling_mass = 0.0
            for i in range(n):
                out_deg = len(out_neighbors[i])
                if out_deg == 0:
                    dangling_mass += scores[i]
                else:
                    share = damping * scores[i] / out_deg
                    for j in out_neighbors[i]:
                        new_scores[j] += share
            if dangling_mass:
                spread = damping * dangling_mass / n
                for j in range(n):
                    new_scores[j] += spread
            delta = sum(abs(new_scores[i] - scores[i]) for i in range(n))
            scores = new_scores
            if delta < _PAGERANK_TOLERANCE:
                break

        return {nodes[i]: scores[i] for i in range(n)}

    def personalized_pagerank(
        self,
        seed_ids: Sequence[int],
        *,
        edge_types: EdgeFilter = None,
        damping: float = 0.85,
        iterations: int = 20,
    ) -> dict[int, float]:
        if not seed_ids:
            return {}

        allowed = self._normalize_edge_types(edge_types)
        nodes = self._all_concept_ids()
        if not nodes:
            return {}

        valid_seeds = [
            s for s in seed_ids if self.get_concept(s) is not None
        ]
        if not valid_seeds:
            return {}

        index = {node_id: i for i, node_id in enumerate(nodes)}
        seed_set = set(valid_seeds)
        n = len(nodes)
        out_neighbors: list[list[int]] = [[] for _ in range(n)]
        adj = self._load_out_neighbors(allowed)

        for src_id, neighbors in adj.items():
            if src_id not in index:
                continue
            src_i = index[src_id]
            for dst_id, _weight in neighbors:
                if dst_id not in index:
                    continue
                out_neighbors[src_i].append(index[dst_id])

        personalization = [0.0] * n
        seed_mass = 1.0 / len(seed_set)
        for s in seed_set:
            if s in index:
                personalization[index[s]] = seed_mass

        scores = [1.0 / n] * n

        for _ in range(iterations):
            new_scores = [(1.0 - damping) * personalization[j] for j in range(n)]
            dangling_mass = 0.0
            for i in range(n):
                out_deg = len(out_neighbors[i])
                if out_deg == 0:
                    dangling_mass += scores[i]
                else:
                    share = damping * scores[i] / out_deg
                    for j in out_neighbors[i]:
                        new_scores[j] += share
            if dangling_mass:
                for j in range(n):
                    new_scores[j] += damping * dangling_mass * personalization[j]
            delta = sum(abs(new_scores[i] - scores[i]) for i in range(n))
            scores = new_scores
            if delta < _PAGERANK_TOLERANCE:
                break

        return {nodes[i]: scores[i] for i in range(n)}

    def vector_search(
        self,
        embedding: Sequence[float],
        *,
        kind: ConceptKind | None = None,
        limit: int = 20,
    ) -> list[tuple[Concept, float]]:
        if limit < 1:
            return []

        query_vec = self._pad_embedding(embedding)
        assert query_vec is not None

        if self._vector_index_ready:
            try:
                k = max(limit * 4, limit)
                result = self._conn.execute(
                    f"""
                    CALL QUERY_VECTOR_INDEX(
                        'Concept',
                        '{_VECTOR_INDEX_NAME}',
                        $query_vec,
                        $k
                    )
                    RETURN node.id, distance
                    ORDER BY distance
                    """,
                    {'query_vec': query_vec, 'k': k},
                )
                scored: list[tuple[Concept, float]] = []
                for row in result:
                    cid = int(_row_value(row, 0))
                    distance = float(_row_value(row, 1))
                    concept = self.get_concept(cid)
                    if concept is None:
                        continue
                    if kind is not None and concept.kind != kind:
                        continue
                    if concept.embedding is None:
                        continue
                    score = 1.0 - distance
                    scored.append((concept, score))
                    if len(scored) >= limit:
                        break
                if scored:
                    scored.sort(key=lambda item: (-item[1], item[0].id))
                    return scored[:limit]
            except Exception as exc:
                log.debug('vector index query failed (%s); falling back to brute force', exc)

        scored_bf: list[tuple[Concept, float]] = []
        result = self._conn.execute(
            """
            MATCH (c:Concept)
            WHERE c.embedding IS NOT NULL
            RETURN c.id, c.name, c.kind, c.first_seen_at, c.last_seen_at,
                   c.centrality_score, c.embedding, c.definition,
                   c.language_hint, c.original_tokens
            """,
        )
        for row in result:
            concept = concept_from_dict(self._concept_row_to_dict(row))
            if kind is not None and concept.kind != kind:
                continue
            if concept.embedding is None:
                continue
            score = _cosine_similarity(query_vec, concept.embedding)
            scored_bf.append((concept, score))

        scored_bf.sort(key=lambda item: (-item[1], item[0].id))
        return scored_bf[:limit]

    def resolve_alias(self, concept_id: int) -> int:
        if self.get_concept(concept_id) is None:
            return concept_id

        visited: list[int] = []
        current = concept_id
        while True:
            if current in visited:
                cycle = ' -> '.join(str(x) for x in (*visited, current))
                raise RuntimeError(
                    f'alias cycle detected involving concept ids: {cycle}',
                )
            visited.append(current)
            if len(visited) > _ALIAS_CHAIN_LIMIT:
                raise RuntimeError(
                    f'alias chain exceeded limit at concept ids: {" -> ".join(str(x) for x in visited)}',
                )
            result = self._conn.execute(
                """
                MATCH (a:Concept {id: $cid})-[:IsCanonicalAliasOf]->(b:Concept)
                RETURN b.id
                LIMIT 1
                """,
                {'cid': current},
            )
            next_id: int | None = None
            for row in result:
                next_id = int(_row_value(row, 0))
                break
            if next_id is None:
                return current
            current = next_id

    def begin_transaction(self) -> GraphTransaction:
        return _KuzuTransaction(self._conn)
