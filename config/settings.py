from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
SETTINGS_PATH = ROOT_DIR / "config" / "platoon_settings.yaml"


def load_settings(path: Path | None = None) -> dict[str, Any]:
    settings_path = path or SETTINGS_PATH
    if not settings_path.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_path}")
    with settings_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Settings file must contain a top-level mapping")
    return data
