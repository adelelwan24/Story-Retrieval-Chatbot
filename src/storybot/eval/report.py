"""Run example questions through the agent and render them as one HTML page.

Each example shows the question, then every step of the trace (Thought, Action,
Observation) and the agent's answer, grouped into tabs by example type. Answers are
trimmed to `max_answer_words` words; observations to `max_obs_chars` characters.
The page is self-contained (inline CSS and JS), so it opens from disk in any browser.
"""
from __future__ import annotations

import html
import json
import time

from .agent_eval import check_case

# example type shown as a tab -> test-case categories it covers
GROUPS = [
    ("Story by id", ["story_by_id", "story_by_id_missing"]),
    ("Story by title", ["story_by_title_exact", "story_by_title_apostrophe", "story_by_title_typo",
                        "story_by_title_partial", "story_by_title_missing"]),
    ("Several stories on a theme", ["theme_search"]),
    ("Theme within a genre", ["genre_search", "genre_search_missing"]),
    ("Find one story", ["find_one"]),
    ("A question about one story", ["ask_about_story", "ask_with_title_typo"]),
    ("Names shared across stories", ["ask_shared_name", "ask_not_in_story"]),
    ("Summaries", ["summarize", "summarize_detailed"]),
    ("Follow-up questions", ["follow_up"]),
    ("Browsing genres", ["list_genres", "list_by_genre"]),
    ("Out of scope", ["out_of_scope"]),
]


def trim_words(text: str, n: int = 500) -> str:
    words = (text or "").split()
    return text.strip() if len(words) <= n else " ".join(words[:n]) + " … [trimmed to %d words]" % n


def run_examples(agent, cases, verbose: bool = True) -> list[dict]:
    """Run each case (all its turns) and keep the full trace of every turn."""
    runs = []
    for case in cases:
        agent.reset()
        t0, turns, last = time.time(), [], None
        try:
            for q in case["turns"]:
                last = agent.run(q)
                turns.append({"question": q, "answer": last.answer,
                              "steps": [{"thought": s.thought, "tool": s.tool, "arguments": s.arguments,
                                         "observation": s.observation} for s in last.steps]})
            checks = check_case(case, last)
            error = None
        except Exception as ex:    # an endpoint error shows on the page instead of stopping the run
            checks, error = {}, f"{type(ex).__name__}: {ex}"
        runs.append({"id": case["id"], "category": case["category"], "turns": turns, "error": error,
                     "reference_answer": case.get("reference_answer", ""), "note": case.get("note", ""),
                     "passed": bool(checks) and all(checks.values()),
                     "failed_checks": [k for k, v in checks.items() if not v],
                     "seconds": round(time.time() - t0, 1)})
        if verbose:
            print(f"{case['id']:10} {'error' if error else 'ok'} {runs[-1]['seconds']}s")
    return runs


CSS = """
:root { --bg:#f3f5f7; --surface:#fff; --ink:#17212b; --muted:#5a6876; --line:#d3dbe3; --accent:#0f6e7a;
  --thought:#6b4fb8; --action:#0f6e7a; --obs:#a3620a; --answer:#2f7a3a; --ref:#5a6876;
  --t-thought:#efeafb; --t-action:#e3f2f3; --t-obs:#fbf0dd; --t-answer:#e6f3e7; --t-ref:#eef1f4;
  --pass:#2f7a3a; --fail:#b3261e; }
@media (prefers-color-scheme: dark) { :root { --bg:#11171d; --surface:#18212a; --ink:#e4eaf0; --muted:#9aa8b5;
  --line:#2c3946; --accent:#4fc0cc; --thought:#b39cf0; --action:#4fc0cc; --obs:#e9a84a; --answer:#7fcb88;
  --ref:#9aa8b5; --t-thought:#251f3a; --t-action:#13303a; --t-obs:#33270f; --t-answer:#17301b;
  --t-ref:#1f2a35; --pass:#7fcb88; --fail:#f2948c; color-scheme: dark; } }
* { box-sizing: border-box; }
[hidden] { display: none !important; }
body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.6 "Segoe UI", system-ui, -apple-system, sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 32px 16px 64px; display: grid; gap: 20px; }
h1 { font-size: 30px; margin: 0; }
.sub { color: var(--muted); margin: 0; }
.mono { font-family: ui-monospace, "Cascadia Code", Consolas, monospace; font-size: 13.5px; }
.tabs { display: flex; flex-wrap: wrap; gap: 8px; }
.tabs button { font: 500 14.5px inherit; font-family: inherit; color: var(--ink); background: var(--surface);
  border: 1px solid var(--line); border-radius: 8px; padding: 9px 15px; cursor: pointer; }
.tabs button[aria-selected="true"] { color: var(--accent); border-color: var(--accent); box-shadow: inset 0 0 0 1px var(--accent); }
.tabs button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.panel { display: grid; gap: 28px; }
.example { display: grid; gap: 8px; }
.head { display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: baseline; color: var(--muted); font-size: 13.5px; }
.badge { font: 600 12px/1 ui-monospace, Consolas, monospace; padding: 4px 8px; border-radius: 999px; border: 1px solid currentColor; }
.pass { color: var(--pass); } .fail { color: var(--fail); }
.q { font-weight: 600; padding: 14px 16px; border: 1px dashed var(--line); border-radius: 10px; background: var(--surface); }
.row { display: grid; grid-template-columns: 130px minmax(0, 1fr); gap: 14px; padding: 12px 16px; border-radius: 10px; }
.row > span { font: 700 12px/1.9 ui-monospace, Consolas, monospace; letter-spacing: .06em; text-transform: uppercase; }
.row > div { min-width: 0; overflow-wrap: anywhere; white-space: pre-wrap; }
.thought { background: var(--t-thought); } .thought > span { color: var(--thought); }
.action { background: var(--t-action); } .action > span { color: var(--action); }
.obs { background: var(--t-obs); } .obs > span { color: var(--obs); }
.answer { background: var(--t-answer); } .answer > span { color: var(--answer); }
.ref { background: var(--t-ref); } .ref > span { color: var(--ref); } .ref > div { color: var(--muted); }
.turn { color: var(--muted); font-size: 13px; margin-top: 6px; }
@media (max-width: 560px) { .row { grid-template-columns: 1fr; gap: 2px; } }
"""

JS = """
const tabs = [...document.querySelectorAll('[role="tab"]')];
function select(t) { tabs.forEach(x => { const on = x === t; x.setAttribute('aria-selected', on);
  x.tabIndex = on ? 0 : -1; document.getElementById(x.getAttribute('aria-controls')).hidden = !on; }); }
tabs.forEach((t, i) => { t.onclick = () => select(t); t.onkeydown = e => {
  const d = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
  if (d) { const n = tabs[(i + d + tabs.length) % tabs.length]; select(n); n.focus(); } }; });
if (tabs.length) select(tabs[0]);
"""


def _row(kind: str, label: str, body: str, mono: bool = False) -> str:
    cls = ' class="mono"' if mono else ""
    return f'<div class="row {kind}"><span>{label}</span><div{cls}>{html.escape(body)}</div></div>'


def _call(tool: str, args: dict) -> str:
    return f"{tool}(" + ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items()) + ")"


def _example(run: dict, max_answer_words: int, max_obs_chars: int, show_reference: bool) -> str:
    parts = ['<article class="example">']
    status = ('<span class="badge pass">checks passed</span>' if run["passed"] else
              f'<span class="badge fail">failed: {html.escape(", ".join(run["failed_checks"]) or "error")}</span>')
    parts.append(f'<div class="head"><span class="mono">{html.escape(run["id"])}</span>{status}'
                 f'<span>{run["seconds"]}s</span></div>')
    for n, turn in enumerate(run["turns"], 1):
        if len(run["turns"]) > 1:
            parts.append(f'<div class="turn">Turn {n} of {len(run["turns"])}</div>')
        parts.append(f'<div class="q">“{html.escape(turn["question"])}”</div>')
        last_thought = None
        for s in turn["steps"]:
            if s["thought"] and s["thought"] != last_thought:   # one thought can cover several calls
                parts.append(_row("thought", "Thought", s["thought"].removeprefix("Thought:").strip()))
                last_thought = s["thought"]
            parts.append(_row("action", "Action", _call(s["tool"], s["arguments"]), mono=True))
            obs = json.dumps(s["observation"], ensure_ascii=False)
            if len(obs) > max_obs_chars:
                obs = obs[:max_obs_chars] + " … [shortened]"
            parts.append(_row("obs", "Observation", obs, mono=True))
        parts.append(_row("answer", "Answer", trim_words(turn["answer"], max_answer_words)))
    if run.get("error"):
        parts.append(_row("obs", "Error", run["error"], mono=True))
    if show_reference and run.get("reference_answer"):
        parts.append(_row("ref", "Expected", run["reference_answer"]))
    parts.append("</article>")
    return "\n".join(parts)


def render_html(runs: list[dict], title: str = "StoryBot agent examples", subtitle: str = "",
                max_answer_words: int = 500, max_obs_chars: int = 600, show_reference: bool = True) -> str:
    by_cat = {}
    for r in runs:
        by_cat.setdefault(r["category"], []).append(r)
    groups = [(name, [r for c in cats for r in by_cat.get(c, [])]) for name, cats in GROUPS]
    known = {c for _, cats in GROUPS for c in cats}
    other = [r for r in runs if r["category"] not in known]
    if other:
        groups.append(("Other", other))
    groups = [(n, rs) for n, rs in groups if rs]

    passed = sum(r["passed"] for r in runs)
    tabs, panels = [], []
    for i, (name, rs) in enumerate(groups):
        tabs.append(f'<button role="tab" id="tab{i}" aria-controls="panel{i}" aria-selected="false">'
                    f'{html.escape(name)} ({len(rs)})</button>')
        body = "\n".join(_example(r, max_answer_words, max_obs_chars, show_reference) for r in rs)
        panels.append(f'<section class="panel" role="tabpanel" id="panel{i}" aria-labelledby="tab{i}">{body}</section>')
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}</style></head>
<body><main>
<h1>{html.escape(title)}</h1>
<p class="sub">{html.escape(subtitle)} {len(runs)} examples, {passed} passed their checks.
Answers are trimmed to {max_answer_words} words and observations are shortened.</p>
<div class="tabs" role="tablist" aria-label="Example types">{''.join(tabs)}</div>
{''.join(panels)}
</main><script>{JS}</script></body></html>"""
