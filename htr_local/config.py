from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("HTR_DATA_DIR", ROOT / "data")).resolve()
DB_PATH = DATA_DIR / "htr.db"
TEMPLATE_PATH = Path(os.getenv("HTR_TEMPLATE", ROOT / "config" / "board_template.json")).resolve()
MODEL_DIR = Path(os.getenv("HTR_MODEL_DIR", DATA_DIR / "model_checkpoints" / "base")).resolve()


@dataclass(frozen=True)
class Field:
    name: str
    kind: str = "text"
    required: bool = False
    allowed: tuple[str, ...] = ()


@dataclass(frozen=True)
class BoardTemplate:
    name: str
    width: int
    height: int
    rows: int
    fields: tuple[Field, ...]
    confidence_threshold: float
    crop_margin: int
    grid_bounds: tuple[float, float, float, float]
    column_bounds: tuple[float, ...]

    @classmethod
    def load(cls, path: Path = TEMPLATE_PATH) -> "BoardTemplate":
        raw = json.loads(path.read_text(encoding="utf-8"))
        fields = tuple(Field(f["name"], f.get("kind", "text"), f.get("required", False), tuple(f.get("allowed", []))) for f in raw["fields"])
        columns = tuple(raw.get("column_bounds", []))
        if columns and len(columns) != len(fields) + 1:
            raise ValueError("column_bounds deve possuir len(fields) + 1 valores")
        return cls(raw["name"], raw["canonical_size"][0], raw["canonical_size"][1], raw["rows"], fields, raw.get("confidence_threshold", 0.70), raw.get("crop_margin", 6), tuple(raw.get("grid_bounds", [0, 0, 1, 1])), columns)


def ensure_directories() -> None:
    for path in (DATA_DIR, DATA_DIR / "raw_images", DATA_DIR / "cells", DATA_DIR / "exports", DATA_DIR / "model_checkpoints"):
        path.mkdir(parents=True, exist_ok=True)
