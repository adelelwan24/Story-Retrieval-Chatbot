"""Task 2 genre classifier: the LoRA adapter on the same server as the chatbot.

The adapter (task2-training/task2_genre_lora_qwen35.ipynb) answers with a genre name. At decode time the
answer is restricted to the label list, so it is always exactly one of the genres:

  vLLM       request field  structured_outputs={"choice": labels}
  llama.cpp  request field  grammar='root ::= "Adventure" | "Alternate Dimension" | ...'

Both build a grammar from the labels and mask every token that would leave it, then decode greedily.
The adapter is chosen per request:

  vLLM       model=<adapter name> (e.g. "genre-lora"), or the base model name for no adapter
  llama.cpp  lora=[{"id": 0, "scale": 1.0}], or scale 0.0 for no adapter
             (the server starts with --lora-init-without-apply, so the chatbot gets the plain base model)

The prompt must match training: prompt.json and labels.json are saved next to the adapter by the notebook.
"""
from __future__ import annotations

import difflib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

BACKENDS = ("vllm", "llamacpp")


@dataclass
class GenrePrediction:
    genre: str          # always one of the labels (closest label when the raw answer was not one)
    raw: str            # what the model wrote
    valid: bool         # raw answer was exactly a label
    seconds: float


def gbnf_choice(labels: list[str]) -> str:
    """GBNF grammar whose only sentences are the labels. JSON string escaping is valid GBNF literal escaping."""
    return "root ::= " + " | ".join(json.dumps(label, ensure_ascii=False) for label in labels) + "\n"


class GenreClassifier:
    def __init__(self, artifacts_dir: str | Path, backend: str = "vllm", base_url: str = "http://localhost:8000/v1",
                 base_model: str = "Qwen/Qwen3.5-4B", adapter_name: str = "genre-lora", lora_id: int = 0,
                 constrain: bool = True, max_tokens: int = 16, timeout: float = 300.0,
                 api_key: str | None = None, ca_bundle: str | None = None, transport=None):
        import httpx
        from storybot.chat.llm import ssl_context

        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        artifacts_dir = Path(artifacts_dir)
        self.labels: list[str] = json.loads((artifacts_dir / "labels.json").read_text(encoding="utf-8"))
        self.prompt: dict = json.loads((artifacts_dir / "prompt.json").read_text(encoding="utf-8"))
        self._label_set = set(self.labels)
        self.grammar = gbnf_choice(self.labels)
        self.backend, self.base_model, self.adapter_name, self.lora_id = backend, base_model, adapter_name, lora_id
        self.constrain, self.max_tokens = constrain, max_tokens

        base_url = base_url.rstrip("/")
        root = base_url[:-3] if base_url.endswith("/v1") else base_url      # /tokenize lives at the server root
        kw = dict(timeout=timeout, headers={"Authorization": f"Bearer {api_key}"} if api_key else {})
        kw.update(transport=transport) if transport else kw.update(verify=ssl_context(ca_bundle))
        self.http = httpx.Client(base_url=root, **kw)

    # ---- prompt ------------------------------------------------------------------------------------
    def _post(self, path: str, body: dict) -> dict:
        r = self.http.post(path, json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code} from {r.request.url}: {r.text[:800]}")
        return r.json()

    def truncate(self, story: str) -> str:
        """Keep the first max_story_tokens tokens, as in training. Uses the server's own tokenizer."""
        limit = self.prompt["max_story_tokens"]
        if self.backend == "vllm":
            ids = self._post("/tokenize", {"model": self.base_model, "prompt": story, "add_special_tokens": False})["tokens"]
            if len(ids) <= limit:
                return story
            return self._post("/detokenize", {"model": self.base_model, "tokens": ids[:limit]})["prompt"]
        ids = self._post("/tokenize", {"content": story, "add_special": False})["tokens"]
        if len(ids) <= limit:
            return story
        return self._post("/detokenize", {"tokens": ids[:limit]})["content"]

    def messages(self, title: str, story: str) -> list[dict]:
        user = self.prompt["user_template"].format(title=title, story=self.truncate(story))
        return [{"role": "system", "content": self.prompt["system_prompt"]}, {"role": "user", "content": user}]

    def request_body(self, messages: list[dict], use_adapter: bool = True, constrain: bool | None = None) -> dict:
        constrain = self.constrain if constrain is None else constrain
        body = {"messages": messages, "temperature": 0.0, "max_tokens": self.max_tokens,
                "chat_template_kwargs": {"enable_thinking": False}}
        if self.backend == "vllm":
            body["model"] = self.adapter_name if use_adapter else self.base_model
            if constrain:
                body["structured_outputs"] = {"choice": self.labels}
        else:
            body["model"] = self.base_model                       # ignored by llama-server
            body["lora"] = [{"id": self.lora_id, "scale": 1.0 if use_adapter else 0.0}]
            body["reasoning_format"] = "none"                     # answer stays in `content`, never parsed as thinking
            if constrain:
                body["grammar"] = self.grammar
        return body

    # ---- prediction --------------------------------------------------------------------------------
    def classify_messages(self, messages: list[dict], use_adapter: bool = True,
                          constrain: bool | None = None) -> GenrePrediction:
        t0 = time.perf_counter()
        msg = self._post("/v1/chat/completions", self.request_body(messages, use_adapter, constrain))["choices"][0]["message"]
        # a server whose reasoning parser misreads the empty <think></think> block may put the answer in reasoning_content
        raw = (msg.get("content") or msg.get("reasoning_content") or "").strip()
        valid = raw in self._label_set
        genre = raw if valid else (difflib.get_close_matches(raw, self.labels, n=1, cutoff=0) or [self.labels[0]])[0]
        return GenrePrediction(genre, raw, valid, time.perf_counter() - t0)

    def predict(self, title: str, story: str, use_adapter: bool = True, constrain: bool | None = None) -> GenrePrediction:
        return self.classify_messages(self.messages(title, story), use_adapter, constrain)

    def predict_many(self, items: list[tuple[str, str]], use_adapter: bool = True, constrain: bool | None = None,
                     workers: int = 8) -> list[GenrePrediction]:
        """(title, story) pairs. Parallel requests let vLLM batch them; llama-server uses its --parallel slots."""
        with ThreadPoolExecutor(max(1, workers)) as pool:
            return list(pool.map(lambda it: self.predict(it[0], it[1], use_adapter, constrain), items))


def build_classifier(cfg, **overrides) -> GenreClassifier:
    """From configs/classify.yaml (`classifier:` section). Keyword overrides win over the YAML."""
    import os

    from storybot.config import resolve

    c = {k: v for k, v in vars(cfg.classifier).items()}
    c.update({k: v for k, v in overrides.items() if v is not None})
    backend = c["backend"]
    return GenreClassifier(resolve(cfg, c["artifacts_dir"]), backend=backend,
                           base_url=os.environ.get("GENRE_BASE_URL") or c[f"{backend}_url"],
                           base_model=c["base_model"], adapter_name=c["adapter_name"], lora_id=c["lora_id"],
                           constrain=c["constrain"], max_tokens=c["max_tokens"], timeout=c["timeout"],
                           api_key=os.environ.get("LLM_API_KEY"), ca_bundle=os.environ.get("LLM_CA_BUNDLE"))
