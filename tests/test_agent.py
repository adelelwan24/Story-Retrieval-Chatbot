"""ReAct loop with a scripted LLM, tool-call parsing, and the OpenAI-compatible backend (mocked HTTP)."""
import json

import httpx

from storybot.chat.agent import StoryAgent
from storybot.chat.llm import AssistantTurn, OpenAICompatLLM, ToolCall, parse_tool_calls
from storybot.chat.tools import SCHEMAS


class FakeTools:
    schemas = SCHEMAS

    def __init__(self):
        self.shown_stories, self.calls = [], []

    def call(self, name, args):
        self.calls.append((name, args))
        if name == "get_story_by_id":
            self.shown_stories.append({"story_id": args["story_id"], "title": "T", "genre": "G", "text": "full"})
        return {"tool": name, "ok": True}


class ScriptedLLM:
    def __init__(self, turns):
        self.turns, self.seen = list(turns), []

    def chat(self, messages, tools=None):
        self.seen.append((list(messages), tools))
        return self.turns.pop(0)


def test_parse_qwen_tool_calls():
    t = parse_tool_calls('Thought: need stories.\n<tool_call>\n{"name": "search_stories", '
                         '"arguments": {"query": "pirates", "num_stories": 6}}\n</tool_call>')
    assert t.content == "Thought: need stories." and t.tool_calls[0].name == "search_stories"
    assert t.tool_calls[0].arguments == {"query": "pirates", "num_stories": 6}
    two = parse_tool_calls('<tool_call>{"name": "a", "arguments": {}}</tool_call><tool_call>{"name": "b", '
                           '"arguments": "{\\"x\\": 1}"}</tool_call>')
    assert [c.name for c in two.tool_calls] == ["a", "b"] and two.tool_calls[1].arguments == {"x": 1}
    bare = parse_tool_calls('{"name": "list_genres", "arguments": {}}')
    assert bare.tool_calls[0].name == "list_genres" and bare.content == ""
    think = parse_tool_calls("<think>hmm</think>The answer.")
    assert think.content == "The answer." and not think.tool_calls


def test_react_loop_runs_tools_then_answers():
    llm = ScriptedLLM([
        AssistantTurn("Thought: the user gave an id.", [ToolCall("get_story_by_id", {"story_id": 5}, "c1")]),
        AssistantTurn("Thought: done.\nT (id 5, G) is shown below."),
    ])
    agent = StoryAgent(llm, FakeTools())
    res = agent.run("show story 5")
    assert res.answer == "T (id 5, G) is shown below."
    assert [(s.tool, s.arguments) for s in res.steps] == [("get_story_by_id", {"story_id": 5})]
    assert res.steps[0].thought.startswith("Thought:") and res.stories[0]["story_id"] == 5
    msgs, tools_given = llm.seen[1]
    assert tools_given == SCHEMAS
    assert msgs[-2]["tool_calls"][0]["function"]["name"] == "get_story_by_id"
    assert msgs[-1] == {"role": "tool", "tool_call_id": "c1", "name": "get_story_by_id",
                        "content": json.dumps({"tool": "get_story_by_id", "ok": True})}
    assert "[1] Action: get_story_by_id" in res.trace()
    # the next turn sees the previous question and answer, without tool traffic
    llm.turns.append(AssistantTurn("It is about T."))
    agent.run("summarize it")
    hist = [m["role"] for m in llm.seen[-1][0]]
    assert hist == ["system", "user", "assistant", "user"]


def test_repeated_call_not_rerun_and_step_budget():
    call = ToolCall("search_stories", {"query": "x"})
    llm = ScriptedLLM([AssistantTurn("", [call]), AssistantTurn("", [ToolCall("search_stories", {"query": "x"})]),
                       AssistantTurn("Final from gathered results.")])
    tools = FakeTools()
    res = StoryAgent(llm, tools, max_steps=2).run("q")
    assert len(tools.calls) == 1                         # second identical call answered from memory
    assert "already called" in res.steps[1].observation["note"]
    assert llm.seen[-1][1] is None                       # forced final answer has no tools
    assert res.answer == "Final from gathered results."


def test_openai_backend_round_trip():
    sent = []

    def handler(request):
        body = json.loads(request.content)
        sent.append(body)
        if len(sent) == 1:
            msg = {"role": "assistant", "content": None, "tool_calls": [{"id": "x1", "type": "function",
                   "function": {"name": "list_genres", "arguments": "{}"}}]}
        else:
            msg = {"role": "assistant", "content": "There are 100 genres."}
        return httpx.Response(200, json={"choices": [{"message": msg}]})

    llm = OpenAICompatLLM("http://test/v1")
    llm.http = httpx.Client(base_url="http://test/v1", transport=httpx.MockTransport(handler))
    res = StoryAgent(llm, FakeTools()).run("which genres?")
    assert res.answer == "There are 100 genres." and res.steps[0].tool == "list_genres"
    assert sent[0]["tools"][0]["function"]["name"] == "get_story_by_id"
    assistant = [m for m in sent[1]["messages"] if m["role"] == "assistant"][0]
    assert assistant["tool_calls"][0]["function"]["arguments"] == "{}"   # JSON string on the wire


def test_ssl_context_ca_bundle(tmp_path, monkeypatch):
    import ssl

    import pytest

    from storybot.chat.llm import ssl_context

    for v in ("LLM_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        monkeypatch.delenv(v, raising=False)
    ctx = ssl_context()
    assert ctx.verify_mode == ssl.CERT_REQUIRED          # verification stays on by default
    monkeypatch.setenv("LLM_CA_BUNDLE", str(tmp_path / "missing.pem"))
    with pytest.raises(FileNotFoundError):
        ssl_context()
