"""Load configs/*.yaml into a nested namespace and resolve paths against the project root."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # the story-chatbot folder


def _ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
    return obj


def load_config(path: str | Path = PROJECT_ROOT / "configs" / "retrieval.yaml", **overrides) -> SimpleNamespace:
    """Read a YAML config. `overrides` replace top-level keys (tests use root=<tmp dir>)."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    raw.update({k: v for k, v in overrides.items() if v is not None})
    # data.*_path entries become absolute here, so code that reads them directly
    # (e.g. cfg.data.metadata_path) does not depend on the working directory
    root = Path(raw.get("root") or PROJECT_ROOT)
    for k, v in (raw.get("data") or {}).items():
        if k.endswith("_path") and v and not Path(v).is_absolute():
            raw["data"][k] = str(root / v)
    return _ns(raw)


def resolve(cfg: SimpleNamespace, rel: str | Path) -> Path:
    """Absolute paths are kept; relative ones are placed under the project root."""
    p = Path(rel)
    return p if p.is_absolute() else Path(getattr(cfg, "root", None) or PROJECT_ROOT) / p
