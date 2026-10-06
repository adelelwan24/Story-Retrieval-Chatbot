"""Agent tools against a real (local, tmp) Qdrant index built with the tiny embedder and toy BM25."""
import numpy as np
import pandas as pd
import pytest

from storybot.chat.llm import AssistantTurn
from storybot.chat.tools import StoryTools, ToolSettings
from storybot.config import load_config
from storybot.data import prepare_data, save_stories
from storybot.retrieval import MetadataMatcher, StoryRetriever, build_index, open_client

TITLES = [("The Clockmaker's Daughter", "Fantasy"), ("The Clockmaker's Daughter", "Fantasy"),   # -> Vol 2
          ("Echoes of the Red Planet", "Science Fiction"), ("Murder at Willow Lane", "Mystery"),
          ("A Quiet Night", "Romance"), ("The Dragon Amulet", "Fantasy"), ("Signal from the Rift", "Science Fiction"),
          ("The Lighthouse Detective", "Mystery"), ("Heart of the Town", "Romance"), ("Crew of the Last Ship", "Science Fiction"),
          ("The Knight Who Stayed", "Fantasy"), ("Night Train Clues", "Mystery")]


def library(long_ids=(2,)):
    rng = np.random.default_rng(1)
    words = "the a ship detective dragon heart night light quiet town knight space crew amulet rift".split()
    rows = []
    for i, (t, g) in enumerate(TITLES):
        n_sent = 140 if i in long_ids else 8
        sents = [" ".join(rng.choice(words, 9)).capitalize() + "." for _ in range(n_sent)]
        sents[1] = f"Captain Zorvik {i} guarded the vault of story{i}."      # a unique, searchable detail
        story = "\n\n".join(" ".join(sents[k:k + 4]) for k in range(0, n_sent, 4))
        rows.append(dict(id=200 + i, title=t, genre=g, story=story))
    return pd.DataFrame(rows)


class FakeLLM:
    def __init__(self):
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append(messages)
        return AssistantTurn(f"summary #{len(self.calls)}")


@pytest.fixture
def index_cfg(tmp_path, tiny_embedder, toy_sparse):
    cfg = load_config(root=str(tmp_path))
    cfg.chunking.chunk_tokens, cfg.chunking.overlap_tokens, cfg.chunking.whole_story_max = 80, 15, 120
    save_stories(library(), tmp_path / cfg.data.stories_path)
    df = prepare_data(cfg)                       # also writes data/story_metadata.json under tmp_path
    build_index(cfg, df, embedder=tiny_embedder, sparse=toy_sparse)
    return cfg


@pytest.fixture
def tools(index_cfg, tiny_embedder, toy_sparse):
    cfg = index_cfg
    retriever = StoryRetriever.from_config(cfg, embedder=tiny_embedder, sparse=toy_sparse)
    matcher = MetadataMatcher(path=cfg.data.metadata_path)
    return StoryTools(retriever, matcher, FakeLLM(), ToolSettings(full_story_chars=2000, summary_chunk_chars=3000))


def test_get_story_by_id(tools):
    out = tools.get_story_by_id(203)
    assert out["found"] and out["story"]["title"] == "Murder at Willow Lane"
    assert tools.call("get_story_by_id", {"story_id": "203"})["story"]["story_id"] == 203   # string id from the LLM
    assert not tools.get_story_by_id(999)["found"]
    assert [s["story_id"] for s in tools.shown_stories] == [203]


def test_get_story_by_title_fuzzy_uses_original_title(tools):
    out = tools.get_story_by_title("echos of the red plannet")
    assert out["found"] and out["matched_title"] == "Echoes of the Red Planet" and out["story"]["story_id"] == 202
    dup = tools.get_story_by_title("the clockmakers daughter")
    assert dup["story"]["story_id"] == 200
    assert any(c["title"] == "The Clockmaker's Daughter - Vol 2" for c in dup["other_similar_titles"])
    assert tools.get_story_by_title("The Clockmaker's Daughter - Vol 2")["story"]["story_id"] == 201


def test_get_story_by_title_falls_back_to_semantic_search(tools):
    out = tools.get_story_by_title("vault guarded by Zorvik story7")
    assert not out["found"] and out["fallback"] == "semantic_search"
    # the toy dense model is random, so only check the BM25-unique story is among the candidates
    assert 207 in [r["story_id"] for r in out["results"]]


def test_search_returns_distinct_stories_default_six(tools):
    out = tools.search_stories("dragon knight amulet")
    ids = [r["story_id"] for r in out["results"]]
    assert len(ids) == 6 and len(set(ids)) == 6
    assert tools.search_stories("dragon", num_stories=50)["num_results"] == len(TITLES)   # capped at max, distinct


def test_search_genre_fuzzy_filter_and_fallback(tools):
    out = tools.search_stories("ship crew", genre="sci-fi fiction science")
    assert out["genre_filter"] == "Science Fiction"
    assert {r["genre"] for r in out["results"]} == {"Science Fiction"}
    out = tools.search_stories("ship crew", genre="Mistery")
    assert out["genre_filter"] == "Mystery"
    bad = tools.search_stories("ship crew", genre="cooking")
    assert "genre_filter" not in bad and "genre_note" in bad and bad["num_results"] == 6


def test_ask_about_story_stays_in_one_story(tools):
    long = tools.ask_about_story("Who guarded the vault?", title="Echoes of the Red Planett")
    assert long["story_id"] == 202 and long["resolved_by"] == "title" and not long["full_story"]
    assert 0 < len(long["passages"]) <= 4
    short = tools.ask_about_story("Who is Zorvik?", story_id=203)
    assert short["full_story"] and short["passages"][0]["text"].startswith(tools.retriever.get_story(203)["text"][:20])
    # unknown title: closest story by meaning, still a single story
    fb = tools.ask_about_story("what happened?", title="vault of story5")
    assert fb["resolved_by"] == "semantic_search" and "note" in fb
    assert fb["story_id"] not in [c["story_id"] for c in fb["other_candidates"]]
    assert all(p["text"] in tools.retriever.get_story(fb["story_id"])["text"] for p in fb["passages"])


def test_summarize_short_and_long(tools):
    short = tools.summarize_story(story_id=203)
    assert short["summary"] == "summary #1"
    n = len(tools.llm.calls)
    long = tools.summarize_story(title="Echoes of the Red Planet", length="detailed")
    assert len(tools.llm.calls) - n > 2          # parts + merge
    tools.summarize_story(title="Echoes of the Red Planet", length="detailed")
    assert long["summary"] == tools.summarize_story(story_id=202, length="detailed")["summary"]   # cached


def test_list_tools_and_errors(tools):
    assert tools.list_genres()["num_genres"] == 4
    out = tools.list_stories_by_genre("romanse")
    assert out["genre"] == "Romance" and {s["title"] for s in out["stories"]} == {"A Quiet Night", "Heart of the Town"}
    assert tools.list_stories_by_genre("cooking")["fallback"] == "semantic_search"
    assert "unknown tool" in tools.call("delete_everything", {})["error"]
    assert "needs argument" in tools.call("search_stories", {})["error"]


def test_build_agent_end_to_end(index_cfg, tiny_embedder, toy_sparse):
    from storybot.chat import ToolCall, build_agent
    from storybot.config import PROJECT_ROOT

    class Scripted:
        def __init__(self):
            self.turns = [AssistantTurn("Thought: a theme request needs at least 6 stories.",
                                        [ToolCall("search_stories", {"query": "dragon amulet", "genre": "fantasy"})]),
                          AssistantTurn("1. The Dragon Amulet (id 205, Fantasy) ...")]

        def chat(self, messages, tools=None):
            return self.turns.pop(0)

    agent = build_agent(retrieval_cfg=index_cfg, chat_cfg=load_config(PROJECT_ROOT / "configs" / "chat.yaml"),
                        llm=Scripted(), embedder=tiny_embedder, sparse=toy_sparse)
    res = agent.run("fantasy stories with a dragon")
    obs = res.steps[0].observation
    assert obs["genre_filter"] == "Fantasy" and {r["genre"] for r in obs["results"]} == {"Fantasy"}
    assert res.answer.startswith("1. The Dragon Amulet")
