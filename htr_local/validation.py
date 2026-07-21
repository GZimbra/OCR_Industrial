from __future__ import annotations

import re
from datetime import datetime

from .config import Field


def validate_value(field: Field, value: str) -> tuple[str, bool]:
    clean = value.strip()
    if field.required and not clean:
        return clean, False
    if not clean:
        return clean, True
    if field.kind == "integer":
        clean = re.sub(r"[^0-9-]", "", clean)
        return clean, bool(re.fullmatch(r"\d+", clean))
    if field.kind == "date":
        clean = clean.replace("-", "/").replace(".", "/")
        try:
            datetime.strptime(clean, "%d/%m/%Y")
            return clean, True
        except ValueError:
            return clean, False
    if field.kind == "choice":
        match = next((item for item in field.allowed if item.casefold() == clean.casefold()), None)
        return match or clean, match is not None
    return clean, True
