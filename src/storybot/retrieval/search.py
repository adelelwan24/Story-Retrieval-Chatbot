"""Query side of the index: story lookups and hybrid (dense + BM25, RRF) search.

`stories`       full stories, read by id or by exact original title (payload filter).
`story_chunks`  hybrid search, either grouped by story (distinct stories, for browsing a
                theme) or restricted to one story_id (questions about a single story).
"""
from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import models as m

from .qdrant_store import DENSE_LATE, SPARSE, story_point_id


def _eq(key: str, value) -> m.FieldCondition:
    return m.FieldCondition(key=key, match=m.MatchValue(value=value))


def _story(p) -> dict:
    pl = p.payload
    return {"story_id": pl["story_id"], "title": pl["title"], "genre": pl["genre"], "text": pl["full_text"]}


@dataclass
class StoryHit:
    story_id: int | str
    title: str
    genre: str
    score: float
    passages: list[dict]          # best chunks of this story: {"chunk_index", "text", "score"}

    def as_dict(self, snippet_chars: int | None = None) -> dict:
        ps = [{**p, "text": p["text"][:snippet_chars] + ("..." if len(p["text"]) > snippet_chars else "")}
              if snippet_chars else p for p in self.passages]
        return {"story_id": self.story_id, "title": self.title, "genre": self.genre,
                "score": round(self.score, 4), "passages": ps}


class StoryRetriever:
    """Thin layer over the two Qdrant collections.

    `embedder` needs `embed_query(text) -> vector` (it adds the "search_query: " prefix);
    `sparse` needs `embed_query(text) -> (indices, values)`.
    """

    def __init__(self, client, embedder, sparse, chunks: str = "story_chunks", stories: str = "stories",
                 dense_vector: str = DENSE_LATE, prefetch_limit: int = 100):
        self.client, self.embedder, self.sparse = client, embedder, sparse
        self.chunks, self.stories = chunks, stories
        self.dense_vector, self.prefetch_limit = dense_vector, prefetch_limit

    @classmethod
    def from_config(cls, cfg, embedder=None, sparse=None, client=None, **kw):
        from .indexing import open_client
        if embedder is None:
            from .embed import DenseEmbedder
            e = cfg.embedding
            embedder = DenseEmbedder(e.model, fp16=e.fp16, doc_prefix=e.doc_prefix,
                                     query_prefix=e.query_prefix, max_ctx=e.max_ctx)
        if sparse is None:
            from .embed import SparseEncoder
            sparse = SparseEncoder(cfg.sparse.model)
        q = cfg.qdrant
        return cls(client or open_client(cfg), embedder, sparse, q.chunks_collection, q.stories_collection, **kw)

    # ---- direct lookups ------------------------------------------------
    def get_story(self, story_id) -> dict | None:
        sid = int(story_id) if str(story_id).strip().isdigit() else story_id
        recs = self.client.retrieve(self.stories, ids=[story_point_id(sid)], with_payload=True)
        return _story(recs[0]) if recs else None

    def get_story_by_title(self, title: str) -> dict | None:
        """Exact match on the original title (as stored in the payload)."""
        pts, _ = self.client.scroll(self.stories, scroll_filter=m.Filter(must=[_eq("title", title)]),
                                    limit=1, with_payload=True)
        return _story(pts[0]) if pts else None

    def list_stories(self, genre: str | None = None, limit: int = 50) -> list[dict]:
        flt = m.Filter(must=[_eq("genre", genre)]) if genre else None
        pts, _ = self.client.scroll(self.stories, scroll_filter=flt, limit=limit, with_payload=["story_id", "title", "genre"])
        return sorted(({"story_id": p.payload["story_id"], "title": p.payload["title"], "genre": p.payload["genre"]}
                       for p in pts), key=lambda d: str(d["title"]))

    # ---- hybrid search ---------------------------------------------------
    def _prefetch(self, query: str, flt: m.Filter | None, limit: int) -> list[m.Prefetch]:
        dense = self.embedder.embed_query(query)
        dense = dense.tolist() if hasattr(dense, "tolist") else list(dense)
        idx, val = self.sparse.embed_query(query)
        return [m.Prefetch(query=dense, using=self.dense_vector, limit=limit, filter=flt),
                m.Prefetch(query=m.SparseVector(indices=list(idx), values=list(val)), using=SPARSE,
                           limit=limit, filter=flt)]

    def search_stories(self, query: str, n_stories: int = 6, genre: str | None = None,
                       passages_per_story: int = 2) -> list[StoryHit]:
        """Distinct stories ranked by their best chunks (RRF of dense + BM25), grouped by story_id."""
        flt = m.Filter(must=[_eq("genre", genre)]) if genre else None
        res = self.client.query_points_groups(
            self.chunks, prefetch=self._prefetch(query, flt, self.prefetch_limit),
            query=m.FusionQuery(fusion=m.Fusion.RRF), group_by="story_id", limit=n_stories,
            group_size=passages_per_story, with_payload=True)
        out = []
        for g in res.groups:
            first = g.hits[0].payload
            out.append(StoryHit(first["story_id"], first["title"], first["genre"], g.hits[0].score,
                                [{"chunk_index": h.payload["chunk_index"], "text": h.payload["text"],
                                  "score": round(h.score, 4)} for h in g.hits]))
        return out

    def search_in_story(self, story_id, query: str, k: int = 4) -> list[dict]:
        """Best passages of ONE story for `query`, returned in story order."""
        sid = int(story_id) if str(story_id).strip().isdigit() else story_id
        flt = m.Filter(must=[_eq("story_id", sid)])
        pts = self.client.query_points(self.chunks, prefetch=self._prefetch(query, flt, max(k * 4, 20)),
                                       query=m.FusionQuery(fusion=m.Fusion.RRF), limit=k,
                                       with_payload=True).points
        hits = [{"chunk_index": p.payload["chunk_index"], "text": p.payload["text"], "score": round(p.score, 4)}
                for p in pts]
        return sorted(hits, key=lambda h: h["chunk_index"])
