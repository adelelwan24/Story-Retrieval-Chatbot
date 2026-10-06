"""Chat with the story agent in a terminal.

    python scripts/chat.py                       # Qwen3-4B via transformers (configs/chat.yaml)
    python scripts/chat.py --backend openai      # a running llama-server / vLLM / Ollama
    python scripts/chat.py --trace               # also print Thought / Action / Observation steps
    python scripts/chat.py -q "stories about lost treasure"   # one question, then exit

Needs the Qdrant index (scripts/build_index.py) and data/story_metadata.json (scripts/prepare_data.py).
Type 'reset' to clear the conversation, 'quit' to exit.
"""
import argparse

from storybot.chat import build_agent
from storybot.config import PROJECT_ROOT, load_config

class color:
    """
    ANSI escape sequences for terminal text coloring
    https://stackoverflow.com/questions/4842424/list-of-ansi-color-escape-sequences
    """
    
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def show(res, trace: bool) -> None:
    if trace and res.steps:
        print(color.YELLOW, res.trace(), color.END, "\n", sep="")
    print(res.answer)
    for st in res.stories:
        print(f"{color.DARKCYAN}\n--- {st['title']} (id {st['story_id']}, {st['genre']}) ---\n{st['text']}{color.END}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "openai"], default=None)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("-q", "--question", default=None)
    a = ap.parse_args()
    ccfg = load_config(PROJECT_ROOT / "configs" / "chat.yaml")
    if a.backend:
        ccfg.llm.backend = a.backend
    agent = build_agent(chat_cfg=ccfg)
    if a.question:
        show(agent.run(a.question), a.trace)
        raise SystemExit
    while True:
        try:
            q = input(f"\n{color.GREEN}you> {color.END}").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q in ("quit", "exit"):
            break
        if q == "reset":
            agent.reset()
            continue
        if q:
            show(agent.run(q), a.trace)
