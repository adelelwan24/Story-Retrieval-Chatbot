"""Qdrant client and the two collections the chatbot uses.

story_chunks  one point per chunk: named dense vectors `dense_late` and `dense_prefix`,
              sparse vector `sparse` (BM25, IDF applied by Qdrant), payload with
              story_id / title / genre / chunk_index (indexed) and the chunk text.
stories       one point per story, no vectors: full text for parent-document retrieval
              and direct lookups by id or title.
"""
from __future__ import annotations

import uuid
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
from qdrant_client import QdrantClient, models as m

DENSE_LATE, DENSE_PREFIX, SPARSE = "dense_late", "dense_prefix", "sparse"

# Local mode locks its folder, so a second QdrantClient(path=...) in the same process
# fails with "already accessed by another instance". Reuse one client per path.
_CLIENTS: dict[str, QdrantClient] = {}


def get_client(path: str | Path | None = None, url: str | None = None, api_key: str | None = None) -> QdrantClient:
    """Server client when `url` is given, otherwise an on-disk local client at `path`."""
    if url:
        return QdrantClient(url=url, api_key=api_key)
    if path is None:
        raise ValueError("give a storage path (local mode) or a url (server)")
    key = str(Path(path).resolve())
    if key not in _CLIENTS:
        Path(key).mkdir(parents=True, exist_ok=True)
        _CLIENTS[key] = QdrantClient(path=key)
    return _CLIENTS[key]


def close_client(path: str | Path) -> None:
    client = _CLIENTS.pop(str(Path(path).resolve()), None)
    if client is not None:
        client.close()


def _py(x):
    """numpy scalars -> plain Python, so the client can serialise payloads."""
    return x.item() if hasattr(x, "item") else x


def chunk_point_id(story_id, chunk_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{story_id}:{chunk_index}"))


def story_point_id(story_id):
    """Integer ids are used as-is; anything else gets a stable UUID."""
    return int(story_id) if str(story_id).isdigit() else str(uuid.uuid5(uuid.NAMESPACE_URL, f"story:{story_id}"))


def collections_ready(client: QdrantClient, chunks: str, stories: str) -> bool:
    """True when both collections exist and hold points (a finished earlier build)."""
    return all(client.collection_exists(c) and client.count(c, exact=True).count > 0 for c in (chunks, stories))


def create_collections(client: QdrantClient, dim: int, chunks: str = "story_chunks",
                       stories: str = "stories", recreate: bool = False, int_ids: bool = True) -> None:
    """Create both collections and their payload indexes. Existing ones are kept unless recreate=True."""
    for name in (chunks, stories):
        if recreate and client.collection_exists(name):
            client.delete_collection(name)

    # Local (on-disk) mode ignores payload indexes and filters by scanning, which is fine
    # for ~1k stories. The calls are kept so a Qdrant server gets the indexes.
    warnings.filterwarnings("ignore", message="Payload indexes have no effect in the local Qdrant")

    if not client.collection_exists(chunks):
        client.create_collection(
            chunks,
            vectors_config={
                DENSE_LATE: m.VectorParams(size=dim, distance=m.Distance.COSINE),
                DENSE_PREFIX: m.VectorParams(size=dim, distance=m.Distance.COSINE),
            },
            sparse_vectors_config={SPARSE: m.SparseVectorParams(modifier=m.Modifier.IDF)},
        )
        # created before the bulk upload so Qdrant indexes while inserting
        client.create_payload_index(chunks, "genre", m.PayloadSchemaType.KEYWORD)
        client.create_payload_index(chunks, "story_id",
                                    m.PayloadSchemaType.INTEGER if int_ids else m.PayloadSchemaType.KEYWORD)
        client.create_payload_index(chunks, "chunk_index", m.PayloadSchemaType.INTEGER)
        client.create_payload_index(chunks, "title", m.PayloadSchemaType.KEYWORD)

    if not client.collection_exists(stories):
        client.create_collection(stories, vectors_config={})   # payload only
        client.create_payload_index(stories, "genre", m.PayloadSchemaType.KEYWORD)
        client.create_payload_index(stories, "title", m.PayloadSchemaType.KEYWORD)   # exact original-title lookups
        client.create_payload_index(stories, "title_lower", m.PayloadSchemaType.KEYWORD)


def upsert_chunks(client: QdrantClient, collection: str, payloads: Sequence[dict],
                  late: np.ndarray, prefix: np.ndarray,
                  sparse: Sequence[tuple[list[int], list[float]]], batch: int = 256) -> int:
    assert len(payloads) == len(late) == len(prefix) == len(sparse), "vector and payload counts differ"
    points = [
        m.PointStruct(
            id=chunk_point_id(p["story_id"], p["chunk_index"]),
            vector={DENSE_LATE: late[i].tolist(), DENSE_PREFIX: prefix[i].tolist(),
                    SPARSE: m.SparseVector(indices=sparse[i][0], values=sparse[i][1])},
            payload=p,
        )
        for i, p in enumerate(payloads)
    ]
    for i in range(0, len(points), batch):
        client.upsert(collection, points[i:i + batch])
    return len(points)


def upsert_stories(client: QdrantClient, collection: str, df, batch: int = 256) -> int:
    points = [
        m.PointStruct(id=story_point_id(r.id), vector={},
                      payload={"story_id": _py(r.id), "title": r.title, "title_lower": r.title.strip().lower(),
                               "genre": r.genre, "full_text": r.story})
        for r in df.itertuples()
    ]
    for i in range(0, len(points), batch):
        client.upsert(collection, points[i:i + batch])
    return len(points)
