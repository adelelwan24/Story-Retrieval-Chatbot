"""GenreClassifier request format for vLLM and llama-server, with a fake HTTP server (no network, no model)."""
import json

import httpx
import pytest

from storybot.classify.genre import GenreClassifier, gbnf_choice

LABELS = ["Adventure", "Mystery", "Mystery Comedy", "Science Fiction"]
PROMPT = {"system_prompt": "Pick one: " + "; ".join(LABELS), "user_template": "Title: {title}\n\nStory:\n{story}",
          "max_story_tokens": 3, "enable_thinking": False}


@pytest.fixture
def artifacts(tmp_path):
    (tmp_path / "labels.json").write_text(json.dumps(LABELS))
    (tmp_path / "prompt.json").write_text(json.dumps(PROMPT))
    return tmp_path


class FakeServer:
    """Whitespace 'tokenizer' and a chat endpoint that returns a fixed answer; records every request."""

    def __init__(self, answer="Mystery", field="content"):
        self.answer, self.field, self.requests = answer, field, []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append((request.url.path, body))
        path = request.url.path
        if path == "/tokenize":
            text = body.get("prompt", body.get("content"))
            return httpx.Response(200, json={"tokens": text.split()})
        if path == "/detokenize":
            text = " ".join(body["tokens"])
            return httpx.Response(200, json={"prompt": text} if "model" in body else {"content": text})
        if path == "/v1/chat/completions":
            return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", self.field: self.answer}}]})
        return httpx.Response(404, text=path)

    def chat_bodies(self):
        return [b for p, b in self.requests if p == "/v1/chat/completions"]


def make(artifacts, backend, server, **kw):
    return GenreClassifier(artifacts, backend=backend, base_url="http://x:1/v1", transport=httpx.MockTransport(server), **kw)


def test_gbnf_lists_every_label_and_escapes_quotes():
    g = gbnf_choice(["Adventure", 'Say "hi"'])
    assert g == 'root ::= "Adventure" | "Say \\"hi\\""\n'


def test_vllm_request_uses_adapter_name_and_choice(artifacts):
    server = FakeServer()
    clf = make(artifacts, "vllm", server, base_model="Qwen/Qwen3.5-4B", adapter_name="genre-lora")
    pred = clf.predict("T", "one two three four five")
    body = server.chat_bodies()[0]
    assert pred.genre == "Mystery" and pred.valid
    assert body["model"] == "genre-lora"
    assert body["structured_outputs"] == {"choice": LABELS}
    assert body["chat_template_kwargs"] == {"enable_thinking": False} and body["temperature"] == 0.0
    assert "lora" not in body and "grammar" not in body
    # story cut to max_story_tokens with the server's tokenizer, then put in the training template
    assert body["messages"][1]["content"] == "Title: T\n\nStory:\none two three"
    assert body["messages"][0]["content"] == PROMPT["system_prompt"]


def test_vllm_base_model_and_unconstrained(artifacts):
    server = FakeServer()
    clf = make(artifacts, "vllm", server, base_model="Qwen/Qwen3.5-4B")
    clf.predict("T", "short", use_adapter=False, constrain=False)
    body = server.chat_bodies()[0]
    assert body["model"] == "Qwen/Qwen3.5-4B" and "structured_outputs" not in body
    assert [p for p, _ in server.requests] == ["/tokenize", "/v1/chat/completions"]   # short story: no detokenize


def test_llamacpp_request_switches_lora_and_sets_grammar(artifacts):
    server = FakeServer()
    clf = make(artifacts, "llamacpp", server, lora_id=0)
    clf.predict("T", "one two three four")
    clf.predict("T", "one", use_adapter=False)
    on, off = server.chat_bodies()
    assert on["lora"] == [{"id": 0, "scale": 1.0}] and off["lora"] == [{"id": 0, "scale": 0.0}]
    assert on["grammar"] == gbnf_choice(LABELS) and on["reasoning_format"] == "none"
    assert "structured_outputs" not in on
    assert ("/detokenize", {"tokens": ["one", "two", "three"]}) in server.requests


def test_answer_in_reasoning_content_is_used(artifacts):
    clf = make(artifacts, "llamacpp", FakeServer("Mystery Comedy", field="reasoning_content"))
    pred = clf.predict("T", "x")
    assert pred.genre == "Mystery Comedy" and pred.valid


def test_invalid_answer_maps_to_closest_label(artifacts):
    clf = make(artifacts, "vllm", FakeServer("science fictionn"))
    pred = clf.predict("T", "x", constrain=False)
    assert not pred.valid and pred.raw == "science fictionn" and pred.genre in LABELS


def test_predict_many_keeps_order(artifacts):
    server = FakeServer()
    clf = make(artifacts, "vllm", server)
    preds = clf.predict_many([("A", "a"), ("B", "b"), ("C", "c")], workers=3)
    assert len(preds) == 3 and all(p.genre == "Mystery" for p in preds)


def test_unknown_backend(artifacts):
    with pytest.raises(ValueError):
        GenreClassifier(artifacts, backend="ollama")
