"""Build the Qdrant collections once: chunk, embed (late + prefix baseline + BM25), upload.

A manifest written at the very end marks a finished build. A rerun with the manifest
present and matching point counts skips everything, so re-running the build never
re-embeds or wipes a finished index.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..config import resolve
from .chunking import build_chunks
from .qdrant_store import create_collections, get_client, upsert_chunks, upsert_stories

MANIFEST = "build_manifest.json"


def _manifest_path(cfg) -> Path:
    # inside the Qdrant folder (also used for the manifest when url: points at a server)
    return resolve(cfg, cfg.qdrant.path) / MANIFEST


def open_client(cfg):
    if cfg.qdrant.url:
        return get_client(url=cfg.qdrant.url)
    return get_client(path=resolve(cfg, cfg.qdrant.path))


def index_is_built(cfg, client=None) -> bool:
    mp = _manifest_path(cfg)
    if not mp.exists():
        return False
    man = json.loads(mp.read_text())
    client = client or open_client(cfg)
    q = cfg.qdrant
    for name, key in ((q.chunks_collection, "n_chunks"), (q.stories_collection, "n_stories")):
        if not client.collection_exists(name) or client.count(name, exact=True).count != man[key]:
            return False
    return True


def build_index(cfg, df, rebuild: bool = False, embedder=None, sparse=None) -> dict:
    """Build both collections from the stories DataFrame `df`. Returns the manifest."""
    client = open_client(cfg)
    if not rebuild and index_is_built(cfg, client) and \
            json.loads(_manifest_path(cfg).read_text())["n_stories"] == len(df):
        man = json.loads(_manifest_path(cfg).read_text())
        print(f"index already built ({man['n_chunks']} chunks, {man['n_stories']} stories); pass rebuild=True to redo it")
        return man

    if embedder is None:
        from .embed import DenseEmbedder
        e = cfg.embedding
        embedder = DenseEmbedder(e.model, fp16=e.fp16, doc_prefix=e.doc_prefix,
                                 query_prefix=e.query_prefix, max_ctx=e.max_ctx)
    if sparse is None:
        from .embed import SparseEncoder
        sparse = SparseEncoder(cfg.sparse.model)

    t0 = time.time()
    c = cfg.chunking
    chunks = build_chunks(df, embedder.count_tokens, chunk_tokens=c.chunk_tokens,
                          overlap_tokens=c.overlap_tokens, whole_story_max=c.whole_story_max)
    n_tok = embedder.count_tokens(df["story"].tolist())
    print(f"{len(chunks)} chunks from {len(df)} stories ({len(chunks) / len(df):.2f} per story); "
          f"stories over max_ctx: {sum(n > cfg.embedding.max_ctx for n in n_tok)}")

    # 1) late chunking: one encoder pass per story, pooled per chunk span
    t1 = time.time()
    by_story: dict = {}
    for i, ch in enumerate(chunks):
        by_story.setdefault(ch.story_id, []).append(i)
    late = np.zeros((len(chunks), embedder.dim), dtype=np.float32)
    stories = dict(zip(df["id"].map(lambda x: x.item() if hasattr(x, "item") else x), df["story"]))
    for sid, idxs in by_story.items():
        first = chunks[idxs[0]]
        late[idxs] = embedder.late_chunk(first.context, stories[sid], [(chunks[i].start, chunks[i].end) for i in idxs])
    t_late = time.time() - t1

    # 2) baseline: each chunk alone, with doc prefix + title/genre line (kept for the A/B)
    t1 = time.time()
    texts = [ch.context + ch.text for ch in chunks]
    prefix = embedder.embed_documents(texts, batch_size=cfg.embedding.batch_size,
                                      max_length=cfg.embedding.baseline_max_length)
    t_prefix = time.time() - t1

    # 3) BM25 on title/genre line + chunk text, never the model's "search_document: " prefix
    t1 = time.time()
    sp = sparse.embed_documents(texts)
    t_sparse = time.time() - t1

    _manifest_path(cfg).unlink(missing_ok=True)   # an interrupted upload must not look finished
    q = cfg.qdrant
    int_ids = all(isinstance(ch.story_id, int) for ch in chunks)
    create_collections(client, embedder.dim, q.chunks_collection, q.stories_collection,
                       recreate=True, int_ids=int_ids)
    n_chunks = upsert_chunks(client, q.chunks_collection, [ch.payload for ch in chunks], late, prefix, sp,
                             batch=q.upsert_batch)
    n_stories = upsert_stories(client, q.stories_collection, df, batch=q.upsert_batch)

    manifest = {
        "embedding_model": cfg.embedding.model,
        "dim": embedder.dim,
        "chunking": vars(cfg.chunking),
        "n_stories": n_stories,
        "n_chunks": n_chunks,
        "late_chunk_fallbacks": getattr(embedder, "late_fallbacks", 0),
        "seconds": {"late": round(t_late, 1), "prefix": round(t_prefix, 1),
                    "sparse": round(t_sparse, 1), "total": round(time.time() - t0, 1)},
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    mp = _manifest_path(cfg)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))
    return manifest
