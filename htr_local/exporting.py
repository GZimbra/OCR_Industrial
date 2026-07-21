from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import DATA_DIR
from .database import rows_for_board


def records(board_id: int) -> list[dict]:
    result: dict[int, dict] = {}
    for cell in rows_for_board(board_id):
        result.setdefault(cell["row_index"], {})[cell["field_name"]] = cell["corrected_text"] if cell["reviewed"] else cell["prediction"]
    return [result[key] for key in sorted(result)]


def export_board(board_id: int, fmt: str) -> Path:
    output = DATA_DIR / "exports" / f"quadro_{board_id}.{fmt}"
    output.parent.mkdir(parents=True, exist_ok=True)
    data = records(board_id)
    if fmt == "json":
        output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    elif fmt == "csv":
        pd.DataFrame(data).to_csv(output, index=False, encoding="utf-8-sig")
    elif fmt == "xlsx":
        pd.DataFrame(data).to_excel(output, index=False)
    else:
        raise ValueError("Formato inválido")
    return output
