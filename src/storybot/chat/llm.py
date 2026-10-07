"""One chat interface, two backends.

HFChatLLM          transformers model (e.g. Qwen3-4B-Instruct-2507 in 4-bit on a GPU). Tools are passed to the
                   chat template; Qwen3 answers with <tool_call>{"name": ..., "arguments": {...}}</tool_call>.
OpenAICompatLLM    any OpenAI-compatible server: llama.cpp `llama-server --jinja` (the CPU build), vLLM, Ollama.

Messages use the OpenAI shape throughout:
  {"role": "assistant", "content": str, "tool_calls": [{"id", "type": "function",
                                                        "function": {"name", "arguments": dict}}]}
  {"role": "tool", "tool_call_id": str, "name": str, "content": str}
"""
from __future__ import annotations

import json
import os
import re
import ssl
import uuid
from dataclasses import dataclass, field

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S)
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)


@dataclass
class ToolCall:
    name: str
    arguments: dict
    id: str = field(default_factory=lambda: "call_" + uuid.uuid4().hex[:8])

    def as_message_part(self) -> dict:
        return {"id": self.id, "type": "function", "function": {"name": self.name, "arguments": self.arguments}}


@dataclass
class AssistantTurn:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)


def _loads(s):
    if isinstance(s, dict):
        return s
    try:
        return json.loads(s) if s else {}
    except json.JSONDecodeError:
        return {"_raw": s}


def parse_tool_calls(text: str) -> AssistantTurn:
    """Split raw model output into visible content and Hermes/Qwen-style <tool_call> blocks.

    Also accepts a bare JSON object {"name": ..., "arguments": ...} as the whole reply,
    which small models sometimes produce without the tags.
    """
    text = _THINK_RE.sub("", text or "")
    calls = []
    for block in _TOOL_CALL_RE.findall(text):
        obj = _loads(block)
        if "name" in obj:
            calls.append(ToolCall(obj["name"], _loads(obj.get("arguments", obj.get("parameters", {})))))
    content = _TOOL_CALL_RE.sub("", text).strip()
    if not calls and content.startswith("{") and content.endswith("}"):
        obj = _loads(content)
        if isinstance(obj, dict) and "name" in obj and ("arguments" in obj or "parameters" in obj):
            calls.append(ToolCall(obj["name"], _loads(obj.get("arguments", obj.get("parameters")))))
            content = ""
    return AssistantTurn(content, calls)


class HFChatLLM:
    """Local transformers model. `load_in_4bit` uses bitsandbytes NF4 (needs a CUDA GPU)."""

    def __init__(self, model_id: str = "Qwen/Qwen3-4B-Instruct-2507", load_in_4bit: bool = True,
                 max_new_tokens: int = 768, temperature: float = 0.0, model=None, tokenizer=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = tokenizer or AutoTokenizer.from_pretrained(model_id)
        if model is None:
            kw = {"device_map": "auto"}
            if load_in_4bit and torch.cuda.is_available():
                from transformers import BitsAndBytesConfig
                kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16)   # T4 has no bf16
            else:
                kw["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
            model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
        self.model = model.eval()
        self.max_new_tokens, self.temperature = max_new_tokens, temperature

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> AssistantTurn:
        prompt = self.tok.apply_chat_template(messages, tools=tools or None, add_generation_prompt=True,
                                              tokenize=False)
        enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        gen = {"max_new_tokens": self.max_new_tokens, "pad_token_id": self.tok.pad_token_id or self.tok.eos_token_id}
        if self.temperature > 0:
            gen.update(do_sample=True, temperature=self.temperature, top_p=0.8)
        else:
            gen.update(do_sample=False)
        with self.torch.no_grad():
            out = self.model.generate(**enc, **gen)
        text = self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return parse_tool_calls(text)


_CA_ENV_VARS = ("LLM_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE")


def ssl_context(ca_bundle: str | None = None) -> ssl.SSLContext:
    """TLS settings for the LLM endpoint; certificates are always verified.

    1. `ca_bundle` (or the LLM_CA_BUNDLE / SSL_CERT_FILE / REQUESTS_CA_BUNDLE env var): a PEM file with the CA
       that signed the endpoint's certificate (a private endpoint or a company proxy).
    2. Otherwise the operating system's certificate store via `truststore` (Windows / macOS keychain), which
       already trusts company proxy certificates installed by IT. Python's default bundle (certifi) does not.
    3. Without truststore installed: Python's default verification.
    """
    ca = ca_bundle or next((os.environ[v] for v in _CA_ENV_VARS if os.environ.get(v)), None)
    if ca:
        if not os.path.exists(ca):
            raise FileNotFoundError(f"CA bundle not found: {ca}")
        return ssl.create_default_context(cafile=ca)
    try:
        import truststore
        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        return ssl.create_default_context()


class OpenAICompatLLM:
    """POST {base_url}/chat/completions with `tools`; works with llama-server --jinja, vLLM and Ollama."""

    def __init__(self, base_url: str = "http://localhost:8080/v1", model: str = "qwen3-4b-instruct",
                 api_key: str | None = None, max_new_tokens: int = 768, temperature: float = 0.0,
                 timeout: float = 300.0, ca_bundle: str | None = None):
        import httpx   # installed with qdrant-client
        self.http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, verify=ssl_context(ca_bundle),
                                 headers={"Authorization": f"Bearer {api_key}"} if api_key else {})
        self.model, self.max_new_tokens, self.temperature = model, max_new_tokens, temperature

    @staticmethod
    def _wire(messages: list[dict]) -> list[dict]:
        """The OpenAI wire format wants tool-call arguments as JSON strings."""
        out = []
        for msg in messages:
            if msg.get("tool_calls"):
                msg = {**msg, "tool_calls": [
                    {**tc, "function": {**tc["function"], "arguments": json.dumps(tc["function"]["arguments"])
                                        if not isinstance(tc["function"]["arguments"], str)
                                        else tc["function"]["arguments"]}} for tc in msg["tool_calls"]]}
            out.append(msg)
        return out

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> AssistantTurn:
        body = {"model": self.model, "messages": self._wire(messages), "max_tokens": self.max_new_tokens,
                "temperature": self.temperature}
        if tools:
            body["tools"] = tools
        r = self.http.post("/chat/completions", json=body)
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        calls = [ToolCall(tc["function"]["name"], _loads(tc["function"].get("arguments")), tc.get("id") or None)
                 for tc in msg.get("tool_calls") or []]
        for c in calls:
            c.id = c.id or "call_" + uuid.uuid4().hex[:8]
        if calls:
            return AssistantTurn(_THINK_RE.sub("", msg.get("content") or "").strip(), calls)
        return parse_tool_calls(msg.get("content") or "")   # servers that leave <tool_call> text unparsed


def build_llm(cfg_llm):
    """Create the backend named in configs/chat.yaml (llm.backend: hf | openai)."""
    if cfg_llm.backend == "hf":
        return HFChatLLM(cfg_llm.model, load_in_4bit=cfg_llm.load_in_4bit,
                         max_new_tokens=cfg_llm.max_new_tokens, temperature=cfg_llm.temperature)
    if cfg_llm.backend == "openai":
        # env vars win over the YAML, so the endpoint can change without editing configs/chat.yaml
        return OpenAICompatLLM(os.environ.get("LLM_BASE_URL") or cfg_llm.base_url,
                               os.environ.get("LLM_MODEL") or cfg_llm.server_model,
                               api_key=cfg_llm.api_key or os.environ.get("LLM_API_KEY"),
                               max_new_tokens=cfg_llm.max_new_tokens, temperature=cfg_llm.temperature,
                               ca_bundle=getattr(cfg_llm, "ca_bundle", None) or os.environ.get("LLM_CA_BUNDLE"))
    raise ValueError(f"unknown llm.backend {cfg_llm.backend!r} (use hf or openai)")
