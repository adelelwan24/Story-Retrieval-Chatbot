"""Run the agent on eval/agent_test_cases.json and save the results.

    python scripts/eval_agent.py --backend openai              # all cases
    python scripts/eval_agent.py --backend openai --ids ask-15 ask-16
    python scripts/eval_agent.py --backend openai --categories ask_about_story find_one

Writes results/agent_eval.json (every case: checks, tools, answer) and prints a summary.
The case ids and titles come from eval/stories_sample.csv, which is part of the full dataset,
so the cases run against the normal 1k-story index.
"""
import argparse
import json

from storybot.chat import build_agent
from storybot.config import PROJECT_ROOT, load_config, resolve
from storybot.eval import load_cases, run_eval, summarize

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "openai"], default=None)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--out", default="results/agent_eval.json")
    a = ap.parse_args()
    ccfg = load_config(PROJECT_ROOT / "configs" / "chat.yaml")
    if a.backend:
        ccfg.llm.backend = a.backend
    rcfg = load_config()
    agent = build_agent(retrieval_cfg=rcfg, chat_cfg=ccfg)
    rows = run_eval(agent, load_cases(ids=a.ids, categories=a.categories))
    summary = summarize(rows)
    out = resolve(rcfg, a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "cases": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print("saved", out)
