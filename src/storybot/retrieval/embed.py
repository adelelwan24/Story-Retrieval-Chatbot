"""Dense (ModernBERT, late chunking + prefix baseline) and sparse (BM25) encoders."""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)


class DenseEmbedder:
    """Mean-pooled, L2-normalised embeddings from a Hugging Face encoder.

    nomic-ai/modernbert-embed-base expects "search_document: " on documents and
    "search_query: " on queries; both prefixes come from the config.
    """

    def __init__(self, model_name: str = "nomic-ai/modernbert-embed-base", device: str | None = None,
                 fp16: bool = True, doc_prefix: str = "search_document: ",
                 query_prefix: str = "search_query: ", max_ctx: int = 8192,
                 model=None, tokenizer=None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = tokenizer or AutoTokenizer.from_pretrained(model_name)
        model = model or AutoModel.from_pretrained(model_name)   # no trust_remote_code needed
        model = model.to(self.device).eval()
        if fp16 and self.device == "cuda":
            model = model.half()
        self.model = model
        self.dim = model.config.hidden_size
        self.doc_prefix, self.query_prefix, self.max_ctx = doc_prefix, query_prefix, max_ctx
        self.late_fallbacks = 0   # chunks that fell outside max_ctx and were embedded alone

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        return [len(x) for x in self.tok(list(texts), add_special_tokens=False)["input_ids"]]

    def _normalise(self, v):
        return self.torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy()

    def embed_texts(self, texts: Sequence[str], batch_size: int = 16, max_length: int = 1100) -> np.ndarray:
        """Embed each text on its own (texts must already carry any prefix)."""
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = self.tok(list(texts[i:i + batch_size]), padding=True, truncation=True,
                               max_length=max_length, return_tensors="pt").to(self.device)
                h = self.model(**enc).last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                out.append(self._normalise((h * m).sum(1) / m.sum(1)))
        return np.vstack(out) if out else np.zeros((0, self.dim), dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([self.query_prefix + query], max_length=512)[0]

    def embed_documents(self, texts: Sequence[str], **kw) -> np.ndarray:
        return self.embed_texts([self.doc_prefix + t for t in texts], **kw)

    def late_chunk(self, context: str, story: str, spans: Sequence[tuple[int, int]]) -> np.ndarray:
        """Run doc_prefix + context + story through the encoder once and mean-pool per span.

        A token belongs to a span when its character midpoint falls inside it, so tokens
        that carry a leading space are not dropped at chunk boundaries. Spans that lie
        past max_ctx (very long stories) are embedded alone as a fallback.
        """
        head = self.doc_prefix + context
        enc = self.tok(head + story, return_offsets_mapping=True, truncation=True,
                       max_length=self.max_ctx, return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0]
        with self.torch.no_grad():
            h = self.model(**enc.to(self.device)).last_hidden_state[0].float().cpu()
        real = offsets[:, 1] > offsets[:, 0]                       # drops special tokens
        mid = (offsets[:, 0] + offsets[:, 1]).float() / 2 - len(head)  # midpoint in story coords
        vecs = []
        for s, e in spans:
            idx = ((mid >= s) & (mid < e) & real).nonzero(as_tuple=True)[0]
            if len(idx) == 0:
                self.late_fallbacks += 1
                log.warning("span %s-%s is beyond max_ctx; embedding it alone", s, e)
                vecs.append(self.embed_documents([context + story[s:e]])[0])
            else:
                vecs.append(self._normalise(h[idx].mean(0)))
        return np.vstack(vecs)


class SparseEncoder:
    """BM25 sparse vectors from FastEmbed; Qdrant applies IDF on its side (Modifier.IDF)."""

    def __init__(self, model_name: str = "Qdrant/bm25"):
        from fastembed import SparseTextEmbedding
        self.model = SparseTextEmbedding(model_name)

    def embed_documents(self, texts: Sequence[str]) -> list[tuple[list[int], list[float]]]:
        return [(v.indices.tolist(), v.values.tolist()) for v in self.model.embed(list(texts))]

    def embed_query(self, text: str) -> tuple[list[int], list[float]]:
        v = next(iter(self.model.query_embed(text)))
        return v.indices.tolist(), v.values.tolist()
