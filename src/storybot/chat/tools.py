"""The agent's tools: story lookups, hybrid search, in-story questions and summaries.

Every tool returns a JSON-serialisable dict (the observation the LLM reads). Tool errors
are returned as {"error": ...} so the agent can recover instead of crashing.

Title and genre inputs are fuzzy-matched against data/story_metadata.json with the same
normalisation used when that file was written; the Qdrant filters then use the ORIGINAL
title / genre. When nothing passes the threshold, the tools fall back to semantic search.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass

from ..retrieval.matching import MetadataMatcher
from ..retrieval.search import StoryRetriever
from . import prompts


@dataclass
class ToolSettings:
    default_num_stories: int = 6      # theme / browse searches return at least this many stories
    max_num_stories: int = 20
    passages_per_story: int = 2
    snippet_chars: int = 300          # passage preview length in search results
    num_passages: int = 4             # ask_about_story
    full_story_chars: int = 6000      # stories up to this size are given whole to ask_about_story
    max_story_chars: int = 12000      # cap on story text placed in an observation
    summary_chunk_chars: int = 10000  # longer stories are summarised part by part
    list_limit: int = 50

    @classmethod
    def from_config(cls, ns) -> "ToolSettings":
        known = {k: v for k, v in vars(ns).items() if k in cls.__dataclass_fields__}
        return cls(**known)


SCHEMAS = [
    {"type": "function", "function": {
        "name": "get_story_by_id",
        "description": "Fetch one story (title, genre, full text) by its numeric story id.",
        "parameters": {"type": "object", "properties": {
            "story_id": {"type": "integer", "description": "The story id, e.g. 523."}},
            "required": ["story_id"]}}},
    {"type": "function", "function": {
        "name": "get_story_by_title",
        "description": ("Fetch one story (title, genre, full text) by its title. Typos and case are tolerated. "
                        "If no title matches, returns the closest stories by meaning instead."),
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string", "description": "The story title as the user wrote it."}},
            "required": ["title"]}}},
    {"type": "function", "function": {
        "name": "search_stories",
        "description": ("Semantic + keyword search over all stories. Returns DISTINCT stories, each with its best "
                        "matching passages. Use num_stories >= 6 when the user wants several stories on a theme; "
                        "use 3 when looking for one story the user describes but cannot name."),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "What the stories should be about (theme, plot, characters, setting)."},
            "genre": {"type": "string", "description": "Optional genre to restrict to, only when the user names one."},
            "num_stories": {"type": "integer", "description": "How many different stories to return (1-20, default 6)."}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "ask_about_story",
        "description": ("Retrieve the passages of ONE story that answer a question about it (a character, event, "
                        "place, ending...). Results never mix stories. Give story_id when known, else title."),
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string", "description": "The user's question about the story."},
            "story_id": {"type": "integer", "description": "The story id, if known."},
            "title": {"type": "string", "description": "The story title, if the id is not known."}},
            "required": ["question"]}}},
    {"type": "function", "function": {
        "name": "summarize_story",
        "description": "Summarise one story. Give story_id when known, else title.",
        "parameters": {"type": "object", "properties": {
            "story_id": {"type": "integer", "description": "The story id, if known."},
            "title": {"type": "string", "description": "The story title, if the id is not known."},
            "length": {"type": "string", "enum": ["short", "detailed"], "description": "Summary length (default short)."}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "list_genres",
        "description": "List all genres in the library.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "list_stories_by_genre",
        "description": "List the titles and ids of stories in one genre (not ranked; use search_stories for a theme).",
        "parameters": {"type": "object", "properties": {
            "genre": {"type": "string", "description": "The genre name; typos are tolerated."},
            "limit": {"type": "integer", "description": "Maximum number of stories (default 50)."}},
            "required": ["genre"]}}},
]


def _as_id(x):
    if x is None or x == "":
        return None
    s = str(x).strip()
    return int(s) if s.lstrip("-").isdigit() else s


class StoryTools:
    """Holds the retriever, matcher and (for summaries) the LLM; `call()` runs a tool by name."""

    schemas = SCHEMAS

    def __init__(self, retriever: StoryRetriever, matcher: MetadataMatcher, llm=None,
                 settings: ToolSettings | None = None):
        self.retriever, self.matcher, self.llm = retriever, matcher, llm
        self.s = settings or ToolSettings()
        self.shown_stories: list[dict] = []     # full stories fetched in this turn (for the UI)
        self._summaries: dict = {}

    # ---- dispatch ------------------------------------------------------
    @property
    def names(self) -> list[str]:
        return [t["function"]["name"] for t in self.schemas]

    def call(self, name: str, args: dict | None) -> dict:
        if name not in self.names:
            return {"error": f"unknown tool '{name}'. Available: {', '.join(self.names)}"}
        fn = getattr(self, name)
        params = inspect.signature(fn).parameters
        args = {k: v for k, v in (args or {}).items() if k in params and v is not None}
        missing = [p for p, v in params.items() if v.default is inspect.Parameter.empty and p not in args]
        if missing:
            return {"error": f"{name} needs argument(s): {', '.join(missing)}"}
        try:
            return fn(**args)
        except Exception as e:  # observation, not a crash: the agent can retry differently
            return {"error": f"{name} failed: {type(e).__name__}: {e}"}

    # ---- helpers -------------------------------------------------------
    def _story_view(self, story: dict) -> dict:
        text = story["text"]
        cap = self.s.max_story_chars
        out = {"story_id": story["story_id"], "title": story["title"], "genre": story["genre"],
               "text": text[:cap]}
        if len(text) > cap:
            out["text_truncated"] = True
        return out

    def _show(self, story: dict) -> None:
        if all(s["story_id"] != story["story_id"] for s in self.shown_stories):
            self.shown_stories.append(story)

    def _search(self, query, n, genre=None) -> list[dict]:
        hits = self.retriever.search_stories(query, n_stories=n, genre=genre,
                                             passages_per_story=self.s.passages_per_story)
        return [h.as_dict(self.s.snippet_chars) for h in hits]

    def _resolve(self, story_id=None, title=None, fallback_query: str | None = None):
        """Find one story from an id or a (fuzzy) title. Returns (story | None, info)."""
        sid = _as_id(story_id)
        if sid is not None:
            st = self.retriever.get_story(sid)
            if st:
                return st, {"resolved_by": "story_id"}
            if not title:
                return None, {"error": f"No story with id {sid}."}
        if title:
            mt = self.matcher.match_title(title)
            if mt.matched:
                st = self.retriever.get_story_by_title(mt.value)
                if st:
                    return st, {"resolved_by": "title", "matched_title": mt.value, "match_score": round(mt.score, 1)}
            hits = self._search(fallback_query or title, 3)
            if hits:
                st = self.retriever.get_story(hits[0]["story_id"])
                if st:
                    return st, {"resolved_by": "semantic_search",
                                "note": (f"No title matched '{title}' closely enough (best score "
                                         f"{mt.score:.0f}); this is the closest story by meaning."),
                                "other_candidates": [{"story_id": h["story_id"], "title": h["title"]} for h in hits[1:]]}
            return None, {"error": f"No story found for title '{title}'."}
        return None, {"error": "Give a story_id or a title."}

    # ---- tools -----------------------------------------------------------
    def get_story_by_id(self, story_id) -> dict:
        sid = _as_id(story_id)
        st = self.retriever.get_story(sid)
        if not st:
            return {"found": False, "message": f"No story with id {sid}."}
        self._show(st)
        return {"found": True, "story": self._story_view(st)}

    def get_story_by_title(self, title: str) -> dict:
        mt = self.matcher.match_title(title)
        if mt.matched:
            st = self.retriever.get_story_by_title(mt.value)
            if st:
                self._show(st)
                others = [c for c in mt.candidates if c["story_id"] != st["story_id"]
                          and c["score"] >= self.matcher.title_threshold - 10]
                out = {"found": True, "matched_title": mt.value, "match_score": round(mt.score, 1),
                       "story": self._story_view(st)}
                if others:
                    out["other_similar_titles"] = others
                return out
        return {"found": False, "fallback": "semantic_search",
                "message": (f"No story title matched '{title}' (best title score {mt.score:.0f}, "
                            f"threshold {self.matcher.title_threshold:.0f}). Closest stories by meaning:"),
                "results": self._search(title, 5)}

    def search_stories(self, query: str, genre: str | None = None, num_stories: int | None = None) -> dict:
        n = int(num_stories or self.s.default_num_stories)
        n = max(1, min(n, self.s.max_num_stories))
        out = {"query": query}
        genre_value = None
        if genre:
            mg = self.matcher.match_genre(genre)
            if mg.matched:
                genre_value = mg.value
                out["genre_filter"] = mg.value
            else:
                out["genre_note"] = (f"Genre '{genre}' matched no genre in the library (closest: "
                                     f"{', '.join(c['genre'] for c in mg.candidates) or 'none'}); "
                                     "searched all genres by meaning instead.")
                query = f"{query} {genre}"
        out["results"] = self._search(query, n, genre_value)
        out["num_results"] = len(out["results"])
        return out

    def ask_about_story(self, question: str, story_id=None, title=None) -> dict:
        st, info = self._resolve(story_id, title, fallback_query=f"{title or ''} {question}".strip())
        if st is None:
            return info
        out = {"story_id": st["story_id"], "title": st["title"], "genre": st["genre"], **info}
        if len(st["text"]) <= self.s.full_story_chars:
            out["full_story"] = True
            out["passages"] = [{"chunk_index": 0, "text": st["text"]}]
        else:
            out["full_story"] = False
            out["passages"] = self.retriever.search_in_story(st["story_id"], question, k=self.s.num_passages)
        return out

    def summarize_story(self, story_id=None, title=None, length: str = "short") -> dict:
        if self.llm is None:
            return {"error": "summarize_story needs an LLM"}
        st, info = self._resolve(story_id, title)
        if st is None:
            return info
        length = length if length in prompts.LENGTH_HINTS else "short"
        key = (st["story_id"], length)
        if key not in self._summaries:
            self._summaries[key] = self._summarize(st, length)
        return {"story_id": st["story_id"], "title": st["title"], "genre": st["genre"], **info,
                "summary": self._summaries[key]}

    def list_genres(self) -> dict:
        return {"num_genres": len(self.matcher.genres), "genres": self.matcher.genres}

    def list_stories_by_genre(self, genre: str, limit: int | None = None) -> dict:
        mg = self.matcher.match_genre(genre)
        if not mg.matched:
            return {"found": False, "fallback": "semantic_search",
                    "message": (f"Genre '{genre}' matched no genre (closest: "
                                f"{', '.join(c['genre'] for c in mg.candidates) or 'none'}). Stories closest by meaning:"),
                    "results": self._search(genre, self.s.default_num_stories)}
        rows = self.retriever.list_stories(mg.value, limit=int(limit or self.s.list_limit))
        return {"found": True, "genre": mg.value, "num_stories": len(rows), "stories": rows}

    # ---- summarisation ---------------------------------------------------
    def _ask(self, system: str, user: str) -> str:
        return self.llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}]).content.strip()

    def _summarize(self, st: dict, length: str) -> str:
        hint, text = prompts.LENGTH_HINTS[length], st["text"]
        size = self.s.summary_chunk_chars
        if len(text) <= size:
            return self._ask(prompts.SUMMARIZE_SYSTEM, prompts.SUMMARIZE_USER.format(
                title=st["title"], genre=st["genre"], length_hint=hint, text=text))
        # map-reduce: notes per part (split on paragraph breaks), then one summary of the notes
        parts, cur = [], ""
        for para in text.split("\n"):
            if cur and len(cur) + len(para) > size:
                parts.append(cur)
                cur = ""
            cur += para + "\n"
        parts.append(cur)
        notes = [self._ask(prompts.SUMMARIZE_SYSTEM, prompts.SUMMARIZE_PART_USER.format(
            i=i + 1, n=len(parts), title=st["title"], text=p)) for i, p in enumerate(parts)]
        return self._ask(prompts.SUMMARIZE_SYSTEM, prompts.SUMMARIZE_MERGE_USER.format(
            title=st["title"], genre=st["genre"], length_hint=hint,
            text="\n\n".join(f"Part {i + 1}:\n{n}" for i, n in enumerate(notes))))
