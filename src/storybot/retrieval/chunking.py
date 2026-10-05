"""Sentence-aware chunking with token overlap.

Chunks are character spans into the original story, so the chunk text is always an
exact substring. Late chunking relies on this to map token offsets back to chunks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

import pandas as pd

# One unit = a sentence (with closing quote) or a line ending in a paragraph break.
UNIT_RE = re.compile(r'[^.!?\n]+(?:[.!?]+["”\']?|\n+|$)\s*')

TokenCounter = Callable[[Sequence[str]], list[int]]


def units_of(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in UNIT_RE.finditer(text) if m.end() > m.start()]


def chunk_story(text: str, count_tokens: TokenCounter, chunk_tokens: int = 400,
                overlap_tokens: int = 50, whole_story_max: int = 1000) -> list[tuple[int, int]]:
    """Return (char_start, char_end) spans covering the story."""
    if count_tokens([text])[0] <= whole_story_max:
        return [(0, len(text))]
    units = units_of(text)
    sizes = count_tokens([text[s:e] for s, e in units])
    spans, i, n = [], 0, len(units)
    while i < n:
        j, total = i, 0
        while j < n and (total + sizes[j] <= chunk_tokens or j == i):
            total += sizes[j]
            j += 1
        spans.append((units[i][0], units[j - 1][1]))
        if j >= n:
            break
        # restart a few units back so neighbouring chunks share ~overlap_tokens
        back, ov = j, 0
        while back > i + 1 and ov < overlap_tokens:
            back -= 1
            ov += sizes[back]
        i = max(back, i + 1)
    return spans


def context_line(title: str, genre: str) -> str:
    """Title/genre header put in front of every chunk (dense and BM25)."""
    return f"Title: {title} | Genre: {genre}\n"


@dataclass
class Chunk:
    story_id: int | str
    title: str
    genre: str
    chunk_index: int
    start: int
    end: int
    text: str
    context: str = field(repr=False)   # title/genre line, without any model prefix

    @property
    def payload(self) -> dict:
        return {"story_id": self.story_id, "title": self.title, "genre": self.genre,
                "chunk_index": self.chunk_index, "start": self.start, "end": self.end,
                "text": self.text}


def build_chunks(df: pd.DataFrame, count_tokens: TokenCounter, **chunk_kw) -> list[Chunk]:
    chunks = []
    for r in df.itertuples():
        ctx = context_line(r.title, r.genre)
        for ci, (s, e) in enumerate(chunk_story(r.story, count_tokens, **chunk_kw)):
            chunks.append(Chunk(r.id.item() if hasattr(r.id, "item") else r.id, r.title, r.genre, ci, s, e, r.story[s:e].strip(), ctx))
    return chunks
