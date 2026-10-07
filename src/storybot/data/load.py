"""Load the 1k stories dataset from a local copy, or from the Hugging Face Hub the first time."""
from __future__ import annotations

from pathlib import Path

import re
import json
import pandas as pd

from storybot.config import load_config

cfg = load_config()

COLUMNS = ["id", "title", "genre", "story"]
STORY_METADATA_PATH = cfg.data.metadata_path


def _read_local(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix in (".jsonl", ".json"):
        return pd.read_json(path, lines=path.suffix == ".jsonl")
    raise ValueError(f"unsupported file type: {path}")



def _normalize_title(title: str) -> str:
    """Normalize a title for duplicate/similarity detection."""
    title = title.lower()
    title = re.sub(r"[^\w\s]", "", title)  # remove punctuation
    title = re.sub(r"\s+", " ", title).strip()
    return title


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"dataset is missing columns {sorted(missing)}")

    df = df[COLUMNS].copy()

    for c in ("title", "genre", "story"):
        df[c] = df[c].astype(str)

    # Integer ids let us use them directly as Qdrant point ids
    if (
        pd.api.types.is_numeric_dtype(df["id"])
        or df["id"].astype(str).str.fullmatch(r"\d+").all()
    ):
        df["id"] = df["id"].astype(int)

    if df["id"].duplicated().any():
        raise ValueError("duplicate story ids")

    # ------------------------------------------------------------------
    # Make similar/duplicate titles unique.
    # The first occurrence keeps its original title.
    # Subsequent occurrences get " - Vol 2", " - Vol 3", etc.
    # ------------------------------------------------------------------
    normalized_titles = df["title"].map(_normalize_title)

    title_counts = {}

    for idx, normalized_title in normalized_titles.items():
        count = title_counts.get(normalized_title, 0) + 1
        title_counts[normalized_title] = count

        if count > 1:
            df.at[idx, "title"] = f"{df.at[idx, 'title']} - Vol {count}"

    return df.reset_index(drop=True)


def load_stories(local_path: str | Path | None = None,
                 dataset_id: str = "FareedKhan/1k_stories_100_genre",
                 limit: int | None = None) -> pd.DataFrame:
    """Return a DataFrame with columns id, title, genre, story.

    Reads `local_path` when it exists; otherwise downloads `dataset_id` from the Hub.
    """
    if local_path is not None and Path(local_path).exists():
        df = _read_local(Path(local_path))
    else:
        from datasets import load_dataset  # only needed for the first download
        df = load_dataset(dataset_id, split="train").to_pandas()
    df = _clean(df)
    save_story_metadata(df)

    return df.head(limit).reset_index(drop=True) if limit else df


def save_stories(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[COLUMNS].to_parquet(path, index=False)
    return path

def save_story_metadata(
    df: pd.DataFrame,
    path: str = STORY_METADATA_PATH,
):
    """
    Save unique story titles and genres for runtime matching.

    The title_map preserves the story ID, original title, and genre.
    """

    title_map = {}

    for _, row in df.iterrows():
        normalized_title = _normalize_title(row["title"])

        title_map[normalized_title] = {
            "id": int(row["id"]),
            "title": row["title"],
            "genre": row["genre"],
        }

    metadata = {
        "title_map": title_map,
        "genres": sorted(df["genre"].dropna().unique().tolist()),
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )


def load_story_metadata(
    path: str = STORY_METADATA_PATH,
):
    """Load story metadata into runtime-friendly structures."""

    with open(path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return {
        "title_map": metadata["title_map"],
        "genres": set(metadata["genres"]),
    }
