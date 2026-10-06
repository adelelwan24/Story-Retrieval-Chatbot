"""Fuzzy matching of user-typed titles and genres against data/story_metadata.json.

Both sides go through the same normalisation used when the metadata file was written
(`storybot.data.load._normalize_title`: lowercase, no punctuation, single spaces).
A match returns the ORIGINAL title or genre, which is what the Qdrant payload filters use.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz, process

from storybot.data.load import _normalize_title, load_story_metadata

normalize = _normalize_title
_VOL_RE = re.compile(r"\s+vol\s+\d+$")   # "the lost key vol 2" -> "the lost key"


def _score(a: str, b: str, **_) -> float:
    """Whole-string similarity that also forgives word order.

    Partial-substring scorers are avoided on purpose: they make a short query such as
    "the night" match almost any title. Partial titles fall back to semantic search.
    """
    return max(fuzz.ratio(a, b), fuzz.token_sort_ratio(a, b))


@dataclass
class Match:
    value: str | None                 # original title / genre, None when nothing passed the threshold
    score: float                      # 0-100
    story_id: int | str | None = None
    candidates: list[dict] = field(default_factory=list)   # close alternatives, best first

    @property
    def matched(self) -> bool:
        return self.value is not None


class MetadataMatcher:
    """Title and genre lookups over the metadata written by `load_stories`."""

    def __init__(self, metadata: dict | None = None, path: str | None = None,
                 title_threshold: float = 85, genre_threshold: float = 80):
        md = metadata if metadata is not None else load_story_metadata(path) if path else load_story_metadata()
        self.title_map: dict[str, dict] = md["title_map"]
        self.genres: list[str] = sorted(md["genres"])
        self._genre_keys = {normalize(g): g for g in self.genres}
        self.title_threshold, self.genre_threshold = title_threshold, genre_threshold

    # ---- titles -------------------------------------------------------
    def match_title(self, title: str, n_candidates: int = 5) -> Match:
        q = normalize(title or "")
        if not q:
            return Match(None, 0.0)
        hits = process.extract(q, list(self.title_map), scorer=_score, limit=n_candidates)
        cands = [{"title": self.title_map[k]["title"], "story_id": self.title_map[k]["id"],
                  "genre": self.title_map[k]["genre"], "score": round(s, 1)} for k, s, _ in hits]
        # other volumes of the same base title ("X", "X - Vol 2", ...) are always worth showing
        base = _VOL_RE.sub("", q)
        for k, v in self.title_map.items():
            if _VOL_RE.sub("", k) == base and all(c["story_id"] != v["id"] for c in cands):
                cands.append({"title": v["title"], "story_id": v["id"], "genre": v["genre"],
                              "score": round(_score(q, k), 1)})
        if q in self.title_map:
            best = self.title_map[q]
            return Match(best["title"], 100.0, best["id"], cands)
        if hits and hits[0][1] >= self.title_threshold:
            best = self.title_map[hits[0][0]]
            return Match(best["title"], hits[0][1], best["id"], cands)
        return Match(None, hits[0][1] if hits else 0.0, None, cands)

    # ---- genres -------------------------------------------------------
    def match_genre(self, genre: str, n_candidates: int = 3) -> Match:
        q = normalize(genre or "")
        if not q:
            return Match(None, 0.0)
        if q in self._genre_keys:
            return Match(self._genre_keys[q], 100.0)
        hits = process.extract(q, list(self._genre_keys), scorer=_score, limit=n_candidates)
        cands = [{"genre": self._genre_keys[k], "score": round(s, 1)} for k, s, _ in hits]
        if hits and hits[0][1] >= self.genre_threshold:
            return Match(self._genre_keys[hits[0][0]], hits[0][1], None, cands)
        return Match(None, hits[0][1] if hits else 0.0, None, cands)
