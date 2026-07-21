from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .config import DB_PATH, ensure_directories

SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS boards(id INTEGER PRIMARY KEY, filename TEXT NOT NULL, image_path TEXT NOT NULL, template_name TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS cells(id INTEGER PRIMARY KEY, board_id INTEGER NOT NULL REFERENCES boards(id) ON DELETE CASCADE, row_index INTEGER NOT NULL, field_name TEXT NOT NULL, image_path TEXT NOT NULL, prediction TEXT NOT NULL DEFAULT '', confidence REAL NOT NULL DEFAULT 0, corrected_text TEXT, reviewed INTEGER NOT NULL DEFAULT 0, needs_review INTEGER NOT NULL DEFAULT 1, updated_at TEXT NOT NULL, UNIQUE(board_id,row_index,field_name));
CREATE TABLE IF NOT EXISTS model_versions(id INTEGER PRIMARY KEY, version TEXT UNIQUE NOT NULL, checkpoint_path TEXT NOT NULL, cer REAL, wer REAL, train_examples INTEGER NOT NULL, created_at TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 0);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connection(path: Path = DB_PATH):
    ensure_directories()
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    try:
        yield db
        db.commit()
    finally:
        db.close()


def create_board(filename: str, image_path: str, template_name: str) -> int:
    with connection() as db:
        cur = db.execute("INSERT INTO boards(filename,image_path,template_name,created_at) VALUES(?,?,?,?)", (filename, image_path, template_name, now()))
        return int(cur.lastrowid)


def add_cell(board_id: int, row: int, field: str, path: str, prediction: str, confidence: float, needs_review: bool) -> int:
    with connection() as db:
        cur = db.execute("INSERT INTO cells(board_id,row_index,field_name,image_path,prediction,confidence,needs_review,updated_at) VALUES(?,?,?,?,?,?,?,?)", (board_id, row, field, path, prediction, confidence, int(needs_review), now()))
        return int(cur.lastrowid)


def correct_cell(cell_id: int, text: str) -> None:
    with connection() as db:
        if not db.execute("SELECT 1 FROM cells WHERE id=?", (cell_id,)).fetchone():
            raise KeyError(cell_id)
        db.execute("UPDATE cells SET corrected_text=?,reviewed=1,needs_review=0,updated_at=? WHERE id=?", (text.strip(), now(), cell_id))


def rows_for_board(board_id: int):
    with connection() as db:
        return [dict(r) for r in db.execute("SELECT * FROM cells WHERE board_id=? ORDER BY row_index,id", (board_id,))]


def get_cell(cell_id: int):
    with connection() as db:
        row = db.execute("SELECT * FROM cells WHERE id=?", (cell_id,)).fetchone()
        return dict(row) if row else None


def list_boards():
    with connection() as db:
        return [dict(r) for r in db.execute("SELECT b.*,COUNT(c.id) cells,SUM(c.needs_review) pending FROM boards b LEFT JOIN cells c ON c.board_id=b.id GROUP BY b.id ORDER BY b.id DESC")]


def dashboard_stats():
    with connection() as db:
        row = db.execute("""SELECT COUNT(DISTINCT b.id) boards, COUNT(c.id) cells,
            COALESCE(SUM(c.needs_review),0) pending, COALESCE(SUM(c.reviewed),0) reviewed,
            COALESCE(AVG(CASE WHEN c.prediction<>'' THEN c.confidence END),0) confidence
            FROM boards b LEFT JOIN cells c ON c.board_id=b.id""").fetchone()
        return dict(row)


def review_queue(pending_only: bool = True, limit: int = 100, offset: int = 0):
    where = "WHERE c.needs_review=1" if pending_only else ""
    with connection() as db:
        rows = db.execute(f"""SELECT c.*,b.filename,b.created_at FROM cells c JOIN boards b ON b.id=c.board_id
            {where} ORDER BY c.needs_review DESC,b.id DESC,c.row_index,c.id LIMIT ? OFFSET ?""", (limit, offset)).fetchall()
        total = db.execute(f"SELECT COUNT(*) FROM cells c {where}").fetchone()[0]
        return {"items": [dict(row) for row in rows], "total": int(total), "limit": limit, "offset": offset}


def training_examples() -> int:
    with connection() as db:
        return int(db.execute("SELECT COUNT(*) FROM cells WHERE reviewed=1 AND corrected_text<>''").fetchone()[0])


def has_board_filename(filename: str, expected_cells: int = 1) -> bool:
    with connection() as db:
        row = db.execute("SELECT COUNT(c.id) FROM boards b LEFT JOIN cells c ON c.board_id=b.id WHERE b.filename=? GROUP BY b.id ORDER BY b.id DESC LIMIT 1", (filename,)).fetchone()
        return row is not None and int(row[0]) >= expected_cells
