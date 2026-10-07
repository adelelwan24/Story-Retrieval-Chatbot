"""Run the story agent on eval/agent_test_cases.json and check each answer.

Checks are computed from the agent's trace (which tools ran, which stories they returned)
and from the final answer text. A case passes when every hard check passes; `any_story_ids`
is reported separately as a soft retrieval check, because on the full 1k index other
stories can rightly outrank the sample ones.
"""
from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from pathlib import Path

from ..config import PROJECT_ROOT

CASES_PATH = PROJECT_ROOT / "eval" / "agent_test_cases.json"
# phrases an answer uses when something was not found or is out of scope (heuristic)
NOT_FOUND_RE = re.compile(r"\b(not|no|couldn'?t|can'?t|cannot|unable|only|doesn'?t|isn'?t|none)\b", re.I)


def load_cases(path: str | Path = CASES_PATH, ids: list[str] | None = None, categories: list[str] | None = None):
    cases = json.loads(Path(path).read_text(encoding="utf-8"))["cases"]
    if ids:
        cases = [c for c in cases if c["id"] in ids]
    if categories:
        cases = [c for c in cases if c["category"] in categories]
    return cases


def _ids_in(obs: dict) -> list:
    """Story ids an observation points at, best first."""
    out = []
    if isinstance(obs.get("story"), dict):
        out.append(obs["story"].get("story_id"))
    if "story_id" in obs:
        out.append(obs["story_id"])
    for key in ("results", "stories"):
        out += [r.get("story_id") for r in obs.get(key) or [] if isinstance(r, dict)]
    return [i for i in out if i is not None]


def check_case(case: dict, res) -> dict:
    """Return {check_name: bool} for one case, given the AgentResult of its last turn."""
    e, ans = case["expect"], (res.answer or "")
    low = ans.lower()
    steps = res.steps
    tools = [s.tool for s in steps]
    touched = [i for s in steps for i in _ids_in(s.observation)]
    by_tool = defaultdict(list)
    for s in steps:
        by_tool[s.tool].append(s)
    c = {}

    if "tools" in e:
        c["tools"] = (not tools) if e["tools"] == [] else bool(set(tools) & set(e["tools"]))
    if "story_id" in e:
        c["story_id"] = e["story_id"] in touched
    if e.get("single_story"):
        asked = {s.observation.get("story_id") for s in by_tool["ask_about_story"]}
        c["single_story"] = asked == {e["story_id"]}
    if "min_distinct_stories" in e:
        sizes = [len({r["story_id"] for r in s.observation.get("results", [])}) for s in by_tool["search_stories"]]
        c["min_distinct_stories"] = max(sizes, default=0) >= e["min_distinct_stories"]
    if "genre_filter" in e:
        obs = [s.observation for s in by_tool["search_stories"] + by_tool["list_stories_by_genre"]]
        used = {o.get("genre_filter") or (o.get("genre") if o.get("found") else None) for o in obs}
        c["genre_filter"] = (used <= {None}) if e["genre_filter"] is None else e["genre_filter"] in used
    if "title_match" in e:
        obs = [s.observation for s in by_tool["get_story_by_title"]]
        kind = e["title_match"]
        if kind == "exact":
            ok = any(o.get("found") and o.get("match_score") == 100 for o in obs)
        elif kind == "fuzzy":
            ok = any(o.get("found") and o.get("match_score", 100) < 100 for o in obs)
        elif kind == "fallback":
            ok = any(not o.get("found") and e.get("story_id") in _ids_in(o) for o in obs)
        else:   # none
            ok = bool(obs) and all(not o.get("found") for o in obs)
        c["title_match"] = ok
    if "summary_length" in e:
        c["summary_length"] = any(s.arguments.get("length") == e["summary_length"] for s in by_tool["summarize_story"])
    for kw in e.get("answer_all", []):
        c[f"answer has '{kw}'"] = kw.lower() in low
    for kw in e.get("answer_none", []):
        c[f"answer avoids '{kw}'"] = kw.lower() not in low
    if e.get("no_story_claimed") or e.get("says_not_found"):
        c["says not found"] = bool(NOT_FOUND_RE.search(ans))
    return c


def run_case(agent, case: dict) -> dict:
    agent.reset()
    t0 = time.time()
    res = None
    for turn in case["turns"]:
        res = agent.run(turn)
    checks = check_case(case, res)
    soft = None
    if "any_story_ids" in case["expect"]:
        touched = {i for s in res.steps for i in _ids_in(s.observation)}
        soft = bool(touched & set(case["expect"]["any_story_ids"]))
    return {"id": case["id"], "category": case["category"], "question": case["turns"][-1],
            "passed": all(checks.values()), "failed_checks": [k for k, v in checks.items() if not v],
            "expected_story_found": soft, "tools": [(s.tool, s.arguments) for s in res.steps],
            "answer": res.answer, "reference_answer": case.get("reference_answer", ""),
            "seconds": round(time.time() - t0, 1)}


def run_eval(agent, cases=None, verbose: bool = True) -> list[dict]:
    rows = []
    for case in cases if cases is not None else load_cases():
        try:
            row = run_case(agent, case)
        except Exception as ex:   # an endpoint error fails this case, not the whole run
            row = {"id": case["id"], "category": case["category"], "question": case["turns"][-1],
                   "passed": False, "failed_checks": [f"error: {type(ex).__name__}: {ex}"],
                   "expected_story_found": None, "tools": [], "answer": "", "seconds": 0.0}
        rows.append(row)
        if verbose:
            mark = "PASS" if row["passed"] else "FAIL " + ", ".join(row["failed_checks"])
            print(f"{row['id']:10} {mark}")
    return rows


def summarize(rows: list[dict]) -> dict:
    by_cat = defaultdict(lambda: [0, 0])
    for r in rows:
        by_cat[r["category"]][0] += r["passed"]
        by_cat[r["category"]][1] += 1
    soft = [r["expected_story_found"] for r in rows if r["expected_story_found"] is not None]
    return {"passed": sum(r["passed"] for r in rows), "total": len(rows),
            "expected_story_found": f"{sum(soft)}/{len(soft)}",
            "by_category": {k: f"{p}/{n}" for k, (p, n) in sorted(by_cat.items())}}
