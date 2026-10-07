"""The agent test cases agree with the sample data, and the scorer catches right and wrong runs."""
import json

import pandas as pd
import pytest

from storybot.chat import StoryAgent, ToolCall
from storybot.chat.llm import AssistantTurn
from storybot.config import PROJECT_ROOT, load_config
from storybot.data import prepare_data, save_stories
from storybot.data.load import _clean
from storybot.eval import check_case, load_cases, run_case
from storybot.retrieval import MetadataMatcher, build_index
from storybot.retrieval.matching import normalize

SAMPLE = PROJECT_ROOT / "eval" / "stories_sample.csv"


@pytest.fixture(scope="module")
def sample():
    return _clean(pd.read_csv(SAMPLE))


@pytest.fixture(scope="module")
def matcher(sample):
    return MetadataMatcher({"title_map": {normalize(r.title): {"id": int(r.id), "title": r.title, "genre": r.genre}
                                          for r in sample.itertuples()},
                            "genres": sorted(sample.genre.unique())})


def test_case_file_is_consistent_with_the_sample(sample):
    cases = load_cases()
    assert len({c["id"] for c in cases}) == len(cases)
    ids, genres = set(sample.id), set(sample.genre)
    for c in cases:
        e = c["expect"]
        assert c["turns"] and e["tools"] is not None, c["id"]
        assert e.get("story_id") is None or e["story_id"] in ids, c["id"]
        assert set(e.get("any_story_ids", [])) <= ids, c["id"]
        assert e.get("genre_filter") is None or e["genre_filter"] in genres, c["id"]
        if "story_id" in e and "answer_all" in e:      # expected answer words really occur in that story
            text = sample.set_index("id").loc[e["story_id"], "story"].lower()
            for kw in e["answer_all"]:
                if c["category"] not in ("story_by_id", "find_one"):   # those expect title words
                    assert kw.lower() in text, (c["id"], kw)


TITLE_CASES = {"title-02": "The Dreamer's Odyssey", "title-03": "Decption's Dark Veil",
               "title-04": "the enchanted forrest of eldria", "title-05": "The Interstelar Odysey",
               "title-06": "The Neon Heist", "title-07": "Harmony Chronicles",
               "title-08": "The Moon That Forgot To Shine", "title-01": "The Great Oceanic Quake"}


@pytest.mark.parametrize("cid", sorted(TITLE_CASES))
def test_title_match_labels(cid, matcher):
    case = {c["id"]: c for c in load_cases()}[cid]
    e, m = case["expect"], matcher.match_title(TITLE_CASES[cid])
    if e["title_match"] == "exact":
        assert m.score == 100 and m.story_id == e["story_id"]
    elif e["title_match"] == "fuzzy":
        assert m.matched and m.score < 100 and m.story_id == e["story_id"]
    else:                                   # fallback / none: below the threshold
        assert not m.matched


@pytest.mark.parametrize("typed, expected", [("Cyberpnk", "Cyberpunk"), ("espionnage", "Espionage"),
                                             ("historical fiction", "Historical Fiction"),
                                             ("coming-of-age", "Coming-of-Age"), ("cooking show", None)])
def test_genre_match_labels(typed, expected, matcher):
    assert matcher.match_genre(typed).value == expected


class Scripted:
    """Plays a fixed tool call, then a fixed answer."""

    def __init__(self, call, answer):
        self.turns = ([AssistantTurn("Thought: x", [call])] if call else []) + [AssistantTurn(answer)]

    def chat(self, messages, tools=None):
        return self.turns.pop(0)


@pytest.fixture
def sample_tools(tmp_path, tiny_embedder, toy_sparse, sample):
    from storybot.chat.tools import StoryTools
    from storybot.retrieval import StoryRetriever
    cfg = load_config(root=str(tmp_path))
    save_stories(sample, tmp_path / cfg.data.stories_path)
    build_index(cfg, prepare_data(cfg), embedder=tiny_embedder, sparse=toy_sparse)
    return StoryTools(StoryRetriever.from_config(cfg, embedder=tiny_embedder, sparse=toy_sparse),
                      MetadataMatcher(path=cfg.data.metadata_path))


def test_scorer_passes_a_good_run_and_flags_a_wrong_story(sample_tools):
    case = {c["id"]: c for c in load_cases()}["ask-15"]          # Victor in Neon Valley
    good = StoryAgent(Scripted(ToolCall("ask_about_story", {"question": "who is Victor?", "title": "The Nightmare of Neon Valley"}),
                               "Victor is one of Isabella's henchmen who turns against her."), sample_tools)
    row = run_case(good, case)
    assert row["passed"], row["failed_checks"]

    bad = StoryAgent(Scripted(ToolCall("ask_about_story", {"question": "who is Victor?", "title": "The Parting Laughter"}),
                              "Victor is the bandit leader Jack beats with a mop handle."), sample_tools)
    row = run_case(bad, case)
    assert not row["passed"]
    assert {"story_id", "single_story", "answer has 'henchm'", "answer avoids 'bandit'"} <= set(row["failed_checks"])


def test_scorer_title_fallback_and_out_of_scope(sample_tools):
    cases = {c["id"]: c for c in load_cases()}
    a = StoryAgent(Scripted(ToolCall("get_story_by_title", {"title": "The Moon That Forgot To Shine"}),
                            "I could not find a story with that title; the closest ones are listed."), sample_tools)
    assert run_case(a, cases["title-08"])["passed"]
    oos = StoryAgent(Scripted(None, "I can only answer questions about the story library."), sample_tools)
    assert run_case(oos, cases["oos-01"])["passed"]
    tool_happy = StoryAgent(Scripted(ToolCall("search_stories", {"query": "France capital"}), "Paris."), sample_tools)
    assert not run_case(tool_happy, cases["oos-01"])["passed"]


def test_examples_html_renders_trace_and_trims(sample_tools):
    from storybot.eval import render_html, run_examples, trim_words
    assert trim_words("a " * 600, 500).endswith("[trimmed to 500 words]")
    assert len(trim_words("a " * 600, 500).split()) == 500 + 5   # 500 words + "… [trimmed to 500 words]"
    case = {c["id"]: c for c in load_cases()}["ask-15"]
    long = "Victor is one of Isabella's henchmen. " + "word " * 700
    agent = StoryAgent(Scripted(ToolCall("ask_about_story", {"question": "who is Victor?",
                                                             "title": "The Nightmare of Neon Valley"}), long), sample_tools)
    runs = run_examples(agent, [case], verbose=False)
    page = render_html(runs)
    assert "ask_about_story(" in page and "Observation" in page and "Answer" in page and "Expected" in page
    assert "[trimmed to 500 words]" in page and "word " * 600 not in page
    assert "Names shared across stories (1)" in page and "<script>" in page
