"""Create the Qdrant collections (story_chunks, stories) and fill them.

Skips the work when a finished build is already on disk; --rebuild redoes it.

    python scripts/build_index.py              # writes the database to qdrant_data/
    python scripts/build_index.py --limit 50   # quick test on 50 stories
"""
import argparse

from storybot.config import load_config
from storybot.data import prepare_data
from storybot.retrieval import build_index

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="index only the first N stories")
    a = ap.parse_args()
    cfg = load_config(a.config) if a.config else load_config()
    df = prepare_data(cfg, limit=a.limit)
    build_index(cfg, df, rebuild=a.rebuild)
