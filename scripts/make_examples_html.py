"""Run example questions through the agent and write an HTML page of the traces and answers.

    python scripts/make_examples_html.py --base-url http://localhost:8080/v1 --model qwen3-4b-instruct --per-type 2
    python scripts/make_examples_html.py --backend openai                  # all 51 test cases
    python scripts/make_examples_html.py --backend openai --per-type 2     # 2 examples of each type
    python scripts/make_examples_html.py --backend openai --ids ask-15 title-05 theme-01
    python scripts/make_examples_html.py --from-json results/agent_examples_runs.json   # re-render, no LLM calls

Questions come from eval/agent_test_cases.json. Writes results/agent_examples.html (open it in a browser)
and results/agent_examples_runs.json (the raw traces, so the page can be re-rendered without re-running).
Each answer is trimmed to --max-words words (default 500).
"""
import argparse
import json
import os

from storybot.config import PROJECT_ROOT, load_config, resolve
from storybot.eval import load_cases
from storybot.eval.report import GROUPS, render_html, run_examples

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "openai"], default=None)
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint, e.g. http://localhost:8080/v1")
    ap.add_argument("--model", default=None, help="model name the endpoint expects")
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--per-type", type=int, default=None, help="at most N examples per example type")
    ap.add_argument("--max-words", type=int, default=500)
    ap.add_argument("--no-reference", action="store_true", help="hide the expected answer under each example")
    ap.add_argument("--from-json", default=None, help="render saved runs instead of calling the agent")
    ap.add_argument("--out", default="results/agent_examples.html")
    a = ap.parse_args()

    rcfg = load_config()
    out = resolve(rcfg, a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ccfg = load_config(PROJECT_ROOT / "configs" / "chat.yaml")
    if a.backend:
        ccfg.llm.backend = a.backend
    if a.base_url or a.model:            # command-line flags win over env vars and configs/chat.yaml
        ccfg.llm.backend = "openai"
        if a.base_url:
            os.environ["LLM_BASE_URL"] = a.base_url
        if a.model:
            os.environ["LLM_MODEL"] = a.model
    if ccfg.llm.backend == "openai":
        model = os.environ.get("LLM_MODEL") or ccfg.llm.server_model
    else:
        model = ccfg.llm.model

    if a.from_json:
        saved = json.loads(resolve(rcfg, a.from_json).read_text(encoding="utf-8"))
        runs, model = saved["runs"], saved.get("model", model)
    else:
        cases = load_cases(ids=a.ids)
        if a.per_type:
            keep = []
            for _, cats in GROUPS:
                keep += [c for c in cases if c["category"] in cats][:a.per_type]
            cases = keep
        from storybot.chat import build_agent
        agent = build_agent(retrieval_cfg=rcfg, chat_cfg=ccfg)
        runs = run_examples(agent, cases)
        raw = out.with_name(out.stem + "_runs.json")
        raw.write_text(json.dumps({"model": model, "runs": runs}, ensure_ascii=False, indent=1), encoding="utf-8")
        print("saved", raw)

    page = render_html(runs, subtitle=f"Model: {model}.", max_answer_words=a.max_words,
                       show_reference=not a.no_reference)
    out.write_text(page, encoding="utf-8")
    print("saved", out)
