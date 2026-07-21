from __future__ import annotations

import argparse
import json
from pathlib import Path

from htr_local.config import MODEL_DIR, BoardTemplate
from htr_local.database import has_board_filename
from htr_local.recognizer import LocalRecognizer
from htr_local.service import ProcessingService


def main() -> None:
    parser = argparse.ArgumentParser(description="Importa fotos locais para revisão e treinamento")
    parser.add_argument("--source", type=Path, default=Path("material"))
    parser.add_argument("--stub", action="store_true", help="Segmenta sem executar reconhecimento")
    args = parser.parse_args()
    files = sorted(path for path in args.source.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not files: raise SystemExit(f"Nenhuma imagem encontrada em {args.source}")
    recognizer = LocalRecognizer(MODEL_DIR, allow_stub=args.stub)
    service = ProcessingService(BoardTemplate.load(), recognizer)
    completed, failures = 0, []
    for index, path in enumerate(files, 1):
        if has_board_filename(path.name, len(service.template.fields) * service.template.rows):
            print(f"[{index}/{len(files)}] JÁ IMPORTADA {path.name}", flush=True)
            continue
        try:
            result = service.process(path.name, path.read_bytes())
            print(f"[{index}/{len(files)}] quadro={result['board_id']} grade={result['grid_quality']:.2f} {path.name}", flush=True)
            completed += 1
        except Exception as exc:
            failures.append({"path": str(path), "error": str(exc)})
            print(f"[{index}/{len(files)}] ERRO {path.name}: {exc}", flush=True)
    report = Path("data") / "material_import_report.json"
    report.write_text(json.dumps({"total": len(files), "completed": completed, "failures": failures}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Importação concluída: {completed}/{len(files)}. Relatório: {report}")


if __name__ == "__main__": main()
