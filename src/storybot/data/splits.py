"""Stratified train/val/test split shared by Task 1 and Task 2.

make_split is the same algorithm as the Task 2 training notebook, so an existing
splits.json and a fresh one made here agree for the same seed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def make_split(df: pd.DataFrame, seed: int = 42, val_frac: float = 0.1, test_frac: float = 0.2) -> dict:
    rng = np.random.default_rng(seed)
    split = {"train": [], "val": [], "test": []}
    for _, grp in df.groupby("genre"):
        ids = grp["id"].tolist()
        rng.shuffle(ids)
        n_test = max(1, round(test_frac * len(ids)))
        n_val = max(1, round(val_frac * len(ids)))
        split["test"] += ids[:n_test]
        split["val"] += ids[n_test:n_test + n_val]
        split["train"] += ids[n_test + n_val:]
    return split


def load_or_create_split(df: pd.DataFrame, path: str | Path, **kw) -> tuple[dict, bool]:
    """Load the frozen split at `path`, or create and save it. Returns (split, created)."""
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text()), False
    split = make_split(df, **kw)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=1))
    return split, True


def attach_split(df: pd.DataFrame, split: dict) -> pd.DataFrame:
    """Add a `split` column; fails if any story id is missing from the split file."""
    where = {i: s for s, ids in split.items() for i in ids}
    out = df.assign(split=df["id"].map(where))
    if out["split"].isna().any():
        raise ValueError("some ids are not in splits.json (was it made from a different dataset version?)")
    return out
