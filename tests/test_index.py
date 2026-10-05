import json

import numpy as np
from qdrant_client import models as m

from storybot.config import load_config
from storybot.data import prepare_data, save_stories
from storybot.retrieval import DENSE_LATE, DENSE_PREFIX, SPARSE, build_index, index_is_built, open_client
from storybot.retrieval.qdrant_store import close_client


def cfg_for(tmp_path):
    cfg = load_config(root=str(tmp_path))
    cfg.chunking.chunk_tokens, cfg.chunking.overlap_tokens, cfg.chunking.whole_story_max = 80, 15, 120
    return cfg


def test_late_chunk_pools_only_story_tokens(tiny_embedder, stories):
    e = tiny_embedder
    story = stories.story[0]
    spans = [(0, len(story) // 2), (len(story) // 2, len(story))]
    v = e.late_chunk("Title: x | Genre: y\n", story, spans)
    assert v.shape == (2, e.dim)
    assert np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-5)
    assert not np.allclose(v[0], v[1])
    assert e.late_fallbacks == 0


def test_baseline_not_truncated_at_512(tiny_embedder):
    e = tiny_embedder
    text = "the ship " * 400            # ~800 tokens + prefix
    a = e.embed_documents([text], max_length=1100)
    b = e.embed_documents([text + "dragon " * 50], max_length=1100)
    assert not np.allclose(a, b)        # the tail past 512 tokens changes the vector


def test_build_skip_rebuild_and_search(tmp_path, stories, tiny_embedder, toy_sparse):
    cfg = cfg_for(tmp_path)
    save_stories(stories, tmp_path / cfg.data.stories_path)      # no Hub download in tests
    df = prepare_data(cfg)
    man = build_index(cfg, df, embedder=tiny_embedder, sparse=toy_sparse)
    assert man["n_stories"] == len(stories) and man["n_chunks"] > len(stories)
    assert (tmp_path / "data" / "splits.json").exists()

    client = open_client(cfg)
    assert index_is_built(cfg, client)
    info = client.get_collection("story_chunks")
    assert set(info.config.params.vectors) == {DENSE_LATE, DENSE_PREFIX}
    assert SPARSE in info.config.params.sparse_vectors

    # second call skips: same manifest, no re-embedding
    again = build_index(cfg, df, embedder=None, sparse=None)
    assert again["built_at"] == man["built_at"]

    # a different story count is not treated as finished
    small = build_index(cfg, df.head(3), embedder=tiny_embedder, sparse=toy_sparse)
    assert small["n_stories"] == 3

    # hybrid query with a genre filter, as the chatbot will run it
    build_index(cfg, df, embedder=tiny_embedder, sparse=toy_sparse)
    qd = tiny_embedder.embed_query("detective in a quiet town").tolist()
    qi, qv = toy_sparse.embed_query("detective in a quiet town")
    flt = m.Filter(must=[m.FieldCondition(key="genre", match=m.MatchValue(value="Mystery"))])
    pts = client.query_points("story_chunks", prefetch=[
        m.Prefetch(query=qd, using=DENSE_LATE, limit=20, filter=flt),
        m.Prefetch(query=m.SparseVector(indices=qi, values=qv), using=SPARSE, limit=20, filter=flt)],
        query=m.FusionQuery(fusion=m.Fusion.RRF), limit=5, with_payload=True).points
    assert pts and all(p.payload["genre"] == "Mystery" for p in pts)

    # parent story lookup by integer id
    rec = client.retrieve("stories", ids=[int(stories.id[1])], with_payload=True)
    assert rec[0].payload["full_text"] == stories.story[1]


def test_index_survives_reopen(tmp_path, stories, tiny_embedder, toy_sparse):
    cfg = cfg_for(tmp_path)
    save_stories(stories, tmp_path / cfg.data.stories_path)
    build_index(cfg, prepare_data(cfg), embedder=tiny_embedder, sparse=toy_sparse)
    close_client(tmp_path / cfg.qdrant.path)            # like restarting the process
    assert index_is_built(cfg)
    assert json.loads((tmp_path / "qdrant_data" / "build_manifest.json").read_text())["n_stories"] == len(stories)
