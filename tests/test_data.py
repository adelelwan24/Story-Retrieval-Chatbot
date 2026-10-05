import json

import pandas as pd
import pytest

from storybot.data import attach_split, load_or_create_split, load_stories, make_split, save_stories


def test_roundtrip_and_clean(tmp_path, stories):
    p = save_stories(stories.assign(id=stories.id.astype(str)), tmp_path / "s.parquet")
    df = load_stories(p)
    assert list(df.columns) == ["id", "title", "genre", "story"]
    assert df["id"].dtype.kind == "i"          # digit strings become ints


def test_duplicate_ids_rejected(tmp_path, stories):
    p = save_stories(pd.concat([stories, stories.head(1)]), tmp_path / "s.parquet")
    with pytest.raises(ValueError):
        load_stories(p)


def test_split_is_stratified_frozen_and_complete(tmp_path):
    df = pd.DataFrame({"id": range(1000), "genre": [f"g{i % 100}" for i in range(1000)],
                       "title": "t", "story": "s"})
    split = make_split(df)
    assert [len(split[k]) for k in ("train", "val", "test")] == [700, 100, 200]
    test_genres = df.set_index("id").loc[split["test"], "genre"]
    assert test_genres.nunique() == 100 and (test_genres.value_counts() == 2).all()

    path = tmp_path / "splits.json"
    s1, created = load_or_create_split(df, path)
    assert created and json.loads(path.read_text()) == s1
    s2, created = load_or_create_split(df, path, seed=999)   # existing file wins over a new seed
    assert not created and s2 == s1
    assert attach_split(df, s1)["split"].notna().all()
    with pytest.raises(ValueError):
        attach_split(pd.concat([df, df.head(1).assign(id=5000)]), s1)
