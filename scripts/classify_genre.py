"""Predict the genre of one story through the running server (vLLM or llama-server).

    python scripts/classify_genre.py --id 523                    # a story from the dataset (true genre shown too)
    python scripts/classify_genre.py --file my_story.txt --title "The Last Signal"
    python scripts/classify_genre.py --id 523 --backend llamacpp --base   # base model without the adapter

Server settings come from configs/classify.yaml (GENRE_BASE_URL overrides the URL).
"""
import argparse
from pathlib import Path

from storybot.classify import build_classifier
from storybot.config import PROJECT_ROOT, load_config

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--id", type=int, help="story id from data/stories.parquet")
    src.add_argument("--file", type=Path, help="text file with the story")
    ap.add_argument("--title", default="Untitled")
    ap.add_argument("--backend", choices=["vllm", "llamacpp"], default=None)
    ap.add_argument("--base", action="store_true", help="adapter off (zero-shot base model)")
    ap.add_argument("--free", action="store_true", help="do not restrict the answer to the labels")
    a = ap.parse_args()

    clf = build_classifier(load_config(PROJECT_ROOT / "configs" / "classify.yaml"), backend=a.backend)
    if a.id is not None:
        from storybot.data import load_stories
        rcfg = load_config()
        df = load_stories(rcfg.data.stories_path, rcfg.data.dataset_id)
        row = df[df["id"] == a.id]
        if row.empty:
            raise SystemExit(f"no story with id {a.id}")
        title, story, truth = row.iloc[0]["title"], row.iloc[0]["story"], row.iloc[0]["genre"]
    else:
        title, story, truth = a.title, a.file.read_text(encoding="utf-8"), None

    p = clf.predict(title, story, use_adapter=not a.base, constrain=not a.free)
    print(f"predicted: {p.genre}" + ("" if p.valid else f"  (raw answer {p.raw!r} was not a label)"))
    if truth:
        print(f"true:      {truth}  ->  {'correct' if p.genre == truth else 'wrong'}")
    print(f"{p.seconds:.2f}s")
