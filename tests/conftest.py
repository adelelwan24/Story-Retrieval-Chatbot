"""Tiny offline fixtures: fake stories, a random ModernBERT and a toy BM25.

No network: the real tokenizer/model and the FastEmbed BM25 model are not needed.
"""
import re

import numpy as np
import pandas as pd
import pytest


GENRES = ["Science Fiction", "Mystery", "Fantasy", "Romance"]


def make_stories(n=8, long_every=3):
    rng = np.random.default_rng(0)
    words = "the a ship detective dragon heart night light quiet town knight space crew amulet rift".split()
    rows = []
    for i in range(n):
        n_sent = 120 if i % long_every == 0 else 6        # some stories are long enough to be split
        sents = [" ".join(rng.choice(words, 9)).capitalize() + "." for _ in range(n_sent)]
        story = "\n\n".join(" ".join(sents[k:k + 4]) for k in range(0, n_sent, 4))
        rows.append(dict(id=100 + i, title=f"Story {i}", genre=GENRES[i % len(GENRES)], story=story))
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def metadata_in_tmp(tmp_path, monkeypatch):
    """load_stories() writes story_metadata.json to a default path fixed at import time;
    point it at the test's tmp dir so tests never touch the project's data/ folder."""
    import storybot.data.load as load
    path = str(tmp_path / "data" / "story_metadata.json")
    monkeypatch.setattr(load.save_story_metadata, "__defaults__", (path,))
    monkeypatch.setattr(load.load_story_metadata, "__defaults__", (path,))
    return path


@pytest.fixture
def stories():
    return make_stories()


@pytest.fixture(scope="session")
def tiny_embedder():
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
    from transformers import ModernBertConfig, ModernBertModel, PreTrainedTokenizerFast

    from storybot.retrieval.embed import DenseEmbedder

    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
    words = ("the a ship detective dragon heart night light quiet town knight space crew amulet rift "
             "search_document search_query title genre story science fiction mystery fantasy romance "
             ": | . 0 1 2 3 4 5 6 7 8 9").split()
    for w in words:
        vocab.setdefault(w, len(vocab))
    t = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    t.normalizer = normalizers.Lowercase()
    t.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Whitespace()])
    from tokenizers.processors import TemplateProcessing
    t.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    tok = PreTrainedTokenizerFast(tokenizer_object=t, pad_token="[PAD]", unk_token="[UNK]",
                                  cls_token="[CLS]", sep_token="[SEP]")
    cfg = ModernBertConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                           num_attention_heads=2, max_position_embeddings=8192, pad_token_id=0,
                           bos_token_id=2, eos_token_id=3, cls_token_id=2, sep_token_id=3,
                           global_attn_every_n_layers=1, local_attention=64)
    import torch
    torch.manual_seed(0)
    model = ModernBertModel(cfg)
    return DenseEmbedder(model=model, tokenizer=tok, device="cpu", max_ctx=8192)


class ToySparse:
    """Stand-in for FastEmbed BM25: term counts over hashed tokens."""

    def _vec(self, text):
        counts = {}
        for w in re.findall(r"\w+", text.lower()):
            k = hash(w) % 50000
            counts[k] = counts.get(k, 0) + 1.0
        ks = sorted(counts)
        return ks, [counts[k] for k in ks]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture
def toy_sparse():
    return ToySparse()
