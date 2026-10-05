"""One-time data step: save a local copy of the dataset and the frozen split."""
from __future__ import annotations

from ..config import resolve
from .load import load_stories, save_stories
from .splits import attach_split, load_or_create_split


def prepare_data(cfg, limit: int | None = None):
    """Make sure stories.parquet and splits.json exist under the project root; return the stories with a split column."""
    stories_path = resolve(cfg, cfg.data.stories_path)
    splits_path = resolve(cfg, cfg.data.splits_path)

    had_copy = stories_path.exists()
    df = load_stories(stories_path, cfg.data.dataset_id)
    if not had_copy:
        save_stories(df, stories_path)
        print(f"saved {len(df)} stories to {stories_path}")
    else:
        print(f"loaded {len(df)} stories from {stories_path}")

    split, created = load_or_create_split(df, splits_path, seed=cfg.data.seed,
                                          val_frac=cfg.data.val_frac, test_frac=cfg.data.test_frac)
    print(("created" if created else "loaded frozen"), "split at", splits_path)
    df = attach_split(df, split)
    print(df["split"].value_counts().to_dict(), "|", df["genre"].nunique(), "genres")
    return df.head(limit).reset_index(drop=True) if limit else df
