"""Pluggable vector store with tunable backends.

Backends:
    - memory:   In-memory numpy (default, no dependencies)
    - qdrant:   Qdrant vector database (tunable HNSW)
    - pgvector: PostgreSQL + pgvector extension (tunable HNSW/IVFFlat)
    - milvus:   Milvus vector database (tunable IVF_FLAT/HNSW)

Selection via REEFTWIN_VECTOR_STORE_BACKEND env var.
All backends implement the VectorStore abstract interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from infrastructure.logging import get_logger

logger = get_logger("genai.vector_store")


@dataclass
class Document:
    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: np.ndarray | None = None


@dataclass
class SearchResult:
    document: Document
    score: float


class VectorStore(ABC):
    """Abstract vector store interface."""

    @property
    @abstractmethod
    def backend_name(self) -> str: ...

    @property
    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def add(self, doc: Document) -> None: ...

    @abstractmethod
    def add_batch(self, docs: list[Document]) -> None: ...

    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[SearchResult]: ...

    @abstractmethod
    def get_all_documents(self) -> list[Document]: ...

    @abstractmethod
    def clear(self) -> None: ...


# ---------------------------------------------------------------------------
# In-Memory Backend (default)
# ---------------------------------------------------------------------------

class InMemoryVectorStore(VectorStore):
    """Numpy-based in-memory vector store. Zero dependencies."""

    def __init__(self) -> None:
        self._documents: dict[str, Document] = {}
        self._embeddings: np.ndarray | None = None
        self._doc_ids: list[str] = []

    @property
    def backend_name(self) -> str:
        return "memory"

    @property
    def count(self) -> int:
        return len(self._documents)

    def add(self, doc: Document) -> None:
        if doc.embedding is None:
            raise ValueError(f"Document {doc.id} has no embedding")
        self._documents[doc.id] = doc
        self._rebuild_index()

    def add_batch(self, docs: list[Document]) -> None:
        for doc in docs:
            if doc.embedding is None:
                raise ValueError(f"Document {doc.id} has no embedding")
            self._documents[doc.id] = doc
        self._rebuild_index()
        logger.info("Added %d documents to %s store (total: %d)", len(docs), self.backend_name, self.count)

    def _rebuild_index(self) -> None:
        self._doc_ids = list(self._documents.keys())
        if self._doc_ids:
            self._embeddings = np.array(
                [self._documents[did].embedding for did in self._doc_ids]
            )
        else:
            self._embeddings = None

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[SearchResult]:
        if self._embeddings is None or len(self._doc_ids) == 0:
            return []
        similarities = self._embeddings @ query_embedding
        top_k = min(k, len(self._doc_ids))
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            doc = self._documents[self._doc_ids[idx]]
            results.append(SearchResult(document=doc, score=float(similarities[idx])))
        return results

    def get_all_documents(self) -> list[Document]:
        return list(self._documents.values())

    def clear(self) -> None:
        self._documents.clear()
        self._embeddings = None
        self._doc_ids = []


# ---------------------------------------------------------------------------
# Qdrant Backend
# ---------------------------------------------------------------------------

class QdrantVectorStore(VectorStore):
    """Qdrant vector database backend with tunable HNSW parameters.

    Tuning:
        QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY, QDRANT_COLLECTION, QDRANT_PREFER_GRPC
    """

    def __init__(self) -> None:
        from infrastructure.settings import settings
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            raise ImportError("Install qdrant-client: pip install 'reeftwin[qdrant]'")

        self._client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key or None,
            prefer_grpc=settings.qdrant_prefer_grpc,
        )
        self._collection = settings.qdrant_collection
        self._documents: dict[str, Document] = {}
        self._dim: int | None = None
        self._Distance = Distance
        self._VectorParams = VectorParams
        logger.info("Qdrant store: %s:%d collection=%s", settings.qdrant_host, settings.qdrant_port, self._collection)

    @property
    def backend_name(self) -> str:
        return "qdrant"

    @property
    def count(self) -> int:
        try:
            info = self._client.get_collection(self._collection)
            return info.points_count or 0
        except Exception:
            return 0

    def _ensure_collection(self, dim: int) -> None:
        if self._dim == dim:
            return
        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=self._VectorParams(
                    size=dim,
                    distance=self._Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection %s (dim=%d)", self._collection, dim)
        self._dim = dim

    def add(self, doc: Document) -> None:
        self.add_batch([doc])

    def add_batch(self, docs: list[Document]) -> None:
        from qdrant_client.models import PointStruct
        if not docs or docs[0].embedding is None:
            return
        self._ensure_collection(len(docs[0].embedding))
        points = [
            PointStruct(
                id=hash(doc.id) % (2**63),
                vector=doc.embedding.tolist(),
                payload={"doc_id": doc.id, "content": doc.content, **doc.metadata},
            )
            for doc in docs if doc.embedding is not None
        ]
        self._client.upsert(collection_name=self._collection, points=points)
        for doc in docs:
            self._documents[doc.id] = doc
        logger.info("Added %d documents to Qdrant (total: %d)", len(docs), self.count)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[SearchResult]:
        results = self._client.search(
            collection_name=self._collection,
            query_vector=query_embedding.tolist(),
            limit=k,
        )
        out = []
        for hit in results:
            doc_id = hit.payload.get("doc_id", str(hit.id))
            doc = self._documents.get(doc_id, Document(
                id=doc_id,
                content=hit.payload.get("content", ""),
                metadata={k: v for k, v in hit.payload.items() if k not in ("doc_id", "content")},
            ))
            out.append(SearchResult(document=doc, score=hit.score))
        return out

    def get_all_documents(self) -> list[Document]:
        return list(self._documents.values())

    def clear(self) -> None:
        try:
            self._client.delete_collection(self._collection)
        except Exception:
            pass
        self._documents.clear()
        self._dim = None


# ---------------------------------------------------------------------------
# pgvector Backend
# ---------------------------------------------------------------------------

class PgVectorStore(VectorStore):
    """PostgreSQL + pgvector backend with tunable HNSW/IVFFlat indexes.

    Tuning:
        PGVECTOR_DSN, PGVECTOR_TABLE, PGVECTOR_INDEX_TYPE (hnsw|ivfflat),
        PGVECTOR_HNSW_M, PGVECTOR_HNSW_EF, PGVECTOR_PROBES
    """

    def __init__(self) -> None:
        from infrastructure.settings import settings
        try:
            import psycopg2
        except ImportError:
            raise ImportError("Install psycopg2: pip install 'reeftwin[pgvector]'")

        self._dsn = settings.pgvector_dsn
        self._table = settings.pgvector_table
        self._index_type = settings.pgvector_index_type
        self._hnsw_m = settings.pgvector_hnsw_m
        self._hnsw_ef = settings.pgvector_hnsw_ef
        self._probes = settings.pgvector_probes
        self._conn = psycopg2.connect(self._dsn)
        self._conn.autocommit = True
        self._documents: dict[str, Document] = {}
        self._dim: int | None = None
        self._setup_extension()
        logger.info("pgvector store: %s table=%s index=%s", self._dsn.split("@")[-1], self._table, self._index_type)

    @property
    def backend_name(self) -> str:
        return "pgvector"

    def _setup_extension(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

    def _ensure_table(self, dim: int) -> None:
        if self._dim == dim:
            return
        with self._conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    id TEXT PRIMARY KEY,
                    content TEXT,
                    metadata JSONB DEFAULT '{{}}',
                    embedding vector({dim})
                )
            """)
            # Create tunable index
            idx_name = f"idx_{self._table}_embedding"
            if self._index_type == "hnsw":
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {idx_name} ON {self._table}
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (m = {self._hnsw_m}, ef_construction = {self._hnsw_ef})
                """)
            elif self._index_type == "ivfflat":
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {idx_name} ON {self._table}
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """)
            # Set search probes for IVFFlat
            if self._index_type == "ivfflat":
                cur.execute(f"SET ivfflat.probes = {self._probes}")
        self._dim = dim

    @property
    def count(self) -> int:
        try:
            with self._conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self._table}")
                return cur.fetchone()[0]
        except Exception:
            return 0

    def add(self, doc: Document) -> None:
        self.add_batch([doc])

    def add_batch(self, docs: list[Document]) -> None:
        if not docs or docs[0].embedding is None:
            return
        import json
        self._ensure_table(len(docs[0].embedding))
        with self._conn.cursor() as cur:
            for doc in docs:
                if doc.embedding is None:
                    continue
                cur.execute(
                    f"INSERT INTO {self._table} (id, content, metadata, embedding) "
                    f"VALUES (%s, %s, %s, %s) ON CONFLICT (id) DO UPDATE SET "
                    f"content=EXCLUDED.content, metadata=EXCLUDED.metadata, embedding=EXCLUDED.embedding",
                    (doc.id, doc.content, json.dumps(doc.metadata), doc.embedding.tolist()),
                )
                self._documents[doc.id] = doc
        logger.info("Added %d documents to pgvector (total: %d)", len(docs), self.count)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[SearchResult]:
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT id, content, metadata, 1 - (embedding <=> %s::vector) as similarity "
                f"FROM {self._table} ORDER BY embedding <=> %s::vector LIMIT %s",
                (query_embedding.tolist(), query_embedding.tolist(), k),
            )
            rows = cur.fetchall()
        import json
        results = []
        for row in rows:
            doc_id, content, meta, score = row
            meta = json.loads(meta) if isinstance(meta, str) else (meta or {})
            doc = self._documents.get(doc_id, Document(id=doc_id, content=content, metadata=meta))
            results.append(SearchResult(document=doc, score=float(score)))
        return results

    def get_all_documents(self) -> list[Document]:
        return list(self._documents.values())

    def clear(self) -> None:
        try:
            with self._conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {self._table}")
        except Exception:
            pass
        self._documents.clear()
        self._dim = None


# ---------------------------------------------------------------------------
# Milvus Backend
# ---------------------------------------------------------------------------

class MilvusVectorStore(VectorStore):
    """Milvus vector database backend with tunable index parameters.

    Tuning:
        MILVUS_HOST, MILVUS_PORT, MILVUS_COLLECTION,
        MILVUS_INDEX_TYPE (IVF_FLAT|HNSW|IVF_SQ8),
        MILVUS_NLIST, MILVUS_NPROBE, MILVUS_METRIC_TYPE (COSINE|L2|IP)
    """

    def __init__(self) -> None:
        from infrastructure.settings import settings
        try:
            from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
        except ImportError:
            raise ImportError("Install pymilvus: pip install 'reeftwin[milvus]'")

        self._collection_name = settings.milvus_collection
        self._index_type = settings.milvus_index_type
        self._nlist = settings.milvus_nlist
        self._nprobe = settings.milvus_nprobe
        self._metric_type = settings.milvus_metric_type
        self._documents: dict[str, Document] = {}
        self._dim: int | None = None
        self._collection = None

        connections.connect(host=settings.milvus_host, port=str(settings.milvus_port))
        self._utility = utility
        self._Collection = Collection
        self._FieldSchema = FieldSchema
        self._CollectionSchema = CollectionSchema
        self._DataType = DataType
        logger.info(
            "Milvus store: %s:%d collection=%s index=%s",
            settings.milvus_host, settings.milvus_port, self._collection_name, self._index_type,
        )

    @property
    def backend_name(self) -> str:
        return "milvus"

    @property
    def count(self) -> int:
        if self._collection is not None:
            self._collection.flush()
            return self._collection.num_entities
        return 0

    def _ensure_collection(self, dim: int) -> None:
        if self._dim == dim and self._collection is not None:
            return

        if self._utility.has_collection(self._collection_name):
            self._collection = self._Collection(self._collection_name)
        else:
            fields = [
                self._FieldSchema(name="pk", dtype=self._DataType.VARCHAR, is_primary=True, max_length=128),
                self._FieldSchema(name="content", dtype=self._DataType.VARCHAR, max_length=4096),
                self._FieldSchema(name="embedding", dtype=self._DataType.FLOAT_VECTOR, dim=dim),
            ]
            schema = self._CollectionSchema(fields=fields)
            self._collection = self._Collection(name=self._collection_name, schema=schema)

            # Build tunable index
            index_params = {"metric_type": self._metric_type, "index_type": self._index_type}
            if self._index_type in ("IVF_FLAT", "IVF_SQ8", "IVF_PQ"):
                index_params["params"] = {"nlist": self._nlist}
            elif self._index_type == "HNSW":
                index_params["params"] = {"M": 16, "efConstruction": 200}

            self._collection.create_index(field_name="embedding", index_params=index_params)
            logger.info("Created Milvus collection %s (dim=%d, index=%s)", self._collection_name, dim, self._index_type)

        self._collection.load()
        self._dim = dim

    def add(self, doc: Document) -> None:
        self.add_batch([doc])

    def add_batch(self, docs: list[Document]) -> None:
        if not docs or docs[0].embedding is None:
            return
        self._ensure_collection(len(docs[0].embedding))
        data = [
            [doc.id for doc in docs],
            [doc.content[:4000] for doc in docs],
            [doc.embedding.tolist() for doc in docs if doc.embedding is not None],
        ]
        self._collection.insert(data)
        for doc in docs:
            self._documents[doc.id] = doc
        logger.info("Added %d documents to Milvus (total: %d)", len(docs), self.count)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[SearchResult]:
        if self._collection is None:
            return []

        search_params = {"metric_type": self._metric_type}
        if self._index_type in ("IVF_FLAT", "IVF_SQ8", "IVF_PQ"):
            search_params["params"] = {"nprobe": self._nprobe}
        elif self._index_type == "HNSW":
            search_params["params"] = {"ef": max(k * 2, 64)}

        results = self._collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=k,
            output_fields=["content"],
        )

        out = []
        for hits in results:
            for hit in hits:
                doc_id = str(hit.id)
                doc = self._documents.get(doc_id, Document(
                    id=doc_id,
                    content=hit.entity.get("content", ""),
                ))
                # Milvus returns distance; convert to similarity for COSINE
                score = 1.0 - hit.distance if self._metric_type == "COSINE" else -hit.distance
                out.append(SearchResult(document=doc, score=float(score)))
        return out

    def get_all_documents(self) -> list[Document]:
        return list(self._documents.values())

    def clear(self) -> None:
        if self._utility.has_collection(self._collection_name):
            self._utility.drop_collection(self._collection_name)
        self._documents.clear()
        self._collection = None
        self._dim = None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_vs_instance: VectorStore | None = None


def get_vector_store(backend: str | None = None) -> VectorStore:
    """Factory for creating vector store with the configured backend.

    Selection via REEFTWIN_VECTOR_STORE_BACKEND env var or explicit parameter.
    """
    global _vs_instance
    if _vs_instance is not None and backend is None:
        return _vs_instance

    from infrastructure.settings import settings
    backend = backend or settings.vector_store_backend

    if backend == "memory":
        store = InMemoryVectorStore()
    elif backend == "qdrant":
        store = QdrantVectorStore()
    elif backend == "pgvector":
        store = PgVectorStore()
    elif backend == "milvus":
        store = MilvusVectorStore()
    else:
        raise ValueError(
            f"Unknown vector store backend: {backend!r}. "
            f"Options: memory, qdrant, pgvector, milvus"
        )

    if backend is None or _vs_instance is None:
        _vs_instance = store

    logger.info("Vector store backend: %s", store.backend_name)
    return store
