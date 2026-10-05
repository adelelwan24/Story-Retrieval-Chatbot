"""Save a local copy of the dataset and the frozen train/val/test split.

    python scripts/prepare_data.py      # writes data/stories.parquet and data/splits.json
"""
import argparse

from storybot.config import load_config
from storybot.data import prepare_data

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config) if a.config else load_config()
    prepare_data(cfg)
