from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from .config import MODEL_DIR, BoardTemplate, ensure_directories
from .database import correct_cell, dashboard_stats, get_cell, list_boards, review_queue, rows_for_board, training_examples
from .exporting import export_board
from .recognizer import LocalRecognizer
from .service import ProcessingService

ensure_directories()
app = FastAPI(title="HTR Local", docs_url="/docs" if os.getenv("HTR_DEBUG") == "1" else None)
template = BoardTemplate.load()
service = ProcessingService(template, LocalRecognizer(MODEL_DIR, allow_stub=os.getenv("HTR_ALLOW_STUB") == "1"))
INDEX = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
_training = {"running": False, "message": "Aguardando", "returncode": None}
_training_lock = threading.Lock()


@app.get("/", response_class=HTMLResponse)
def index(): return INDEX


@app.get("/api/status")
def status():
    material = Path(__file__).parent.parent / "material"
    material_total = sum(1 for p in material.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}) if material.exists() else 0
    return {"offline": True, "model_ready": MODEL_DIR.exists(), "model_dir": str(MODEL_DIR), "template": template.name, "training_examples": training_examples(), "training": _training.copy(), "material_total": material_total, "stats": dashboard_stats()}


@app.post("/api/boards")
async def process_board(image: UploadFile = File(...)):
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(415, "Use JPEG, PNG ou WebP")
    content = await image.read()
    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(413, "Imagem excede 25 MB")
    try: return service.process(image.filename or "quadro", content)
    except (ValueError, RuntimeError) as exc: raise HTTPException(422, str(exc)) from exc


@app.get("/api/boards")
def history(): return list_boards()


@app.get("/api/review")
def pending_review(pending_only: bool = True, limit: int = 100, offset: int = 0):
    return review_queue(pending_only, min(max(limit, 1), 250), max(offset, 0))


@app.get("/api/boards/{board_id}")
def board(board_id: int): return rows_for_board(board_id)


@app.get("/api/boards/{board_id}/image")
def board_image(board_id: int):
    board_rows = [item for item in list_boards() if item["id"] == board_id]
    if not board_rows: raise HTTPException(404, "Quadro não encontrado")
    path = Path(board_rows[0]["image_path"]).resolve()
    allowed_roots = [(Path(__file__).parent.parent / "data" / "raw_images").resolve(), (Path(__file__).parent.parent / "material").resolve()]
    if not any(root == path.parent or root in path.parents for root in allowed_roots) or not path.is_file(): raise HTTPException(404, "Imagem não encontrada")
    return FileResponse(path)


@app.put("/api/cells/{cell_id}")
def review(cell_id: int, text: str = Form(...)):
    try: correct_cell(cell_id, text)
    except KeyError: raise HTTPException(404, "Célula não encontrada")
    return {"ok": True}


@app.get("/api/cells/{cell_id}/image")
def cell_image(cell_id: int):
    cell = get_cell(cell_id)
    if not cell: raise HTTPException(404, "Célula não encontrada")
    path = Path(cell["image_path"]).resolve()
    allowed = (Path(__file__).parent.parent / "data" / "cells").resolve()
    if allowed not in path.parents or not path.is_file(): raise HTTPException(404, "Imagem não encontrada")
    return FileResponse(path, media_type="image/png")


@app.get("/api/boards/{board_id}/export/{fmt}")
def export(board_id: int, fmt: str):
    try: path = export_board(board_id, fmt)
    except ValueError as exc: raise HTTPException(400, str(exc)) from exc
    return FileResponse(path, filename=path.name)


def _run_training(epochs: int) -> None:
    env = os.environ.copy(); env["HF_HUB_OFFLINE"] = "1"; env["TRANSFORMERS_OFFLINE"] = "1"
    command = [sys.executable, "-m", "scripts.train", "--epochs", str(epochs)]
    result = subprocess.run(command, cwd=Path(__file__).parent.parent, env=env, capture_output=True, text=True, timeout=24 * 60 * 60)
    message = (result.stdout if result.returncode == 0 else result.stderr)[-4000:]
    with _training_lock: _training.update(running=False, returncode=result.returncode, message=message)


@app.post("/api/train")
def train(epochs: int = Form(3)):
    if not 1 <= epochs <= 50: raise HTTPException(400, "epochs deve estar entre 1 e 50")
    with _training_lock:
        if _training["running"]: raise HTTPException(409, "Treino já está em execução")
        _training.update(running=True, returncode=None, message="Preparando dataset local")
    threading.Thread(target=_run_training, args=(epochs,), daemon=True).start()
    return {"started": True, "examples": training_examples()}
