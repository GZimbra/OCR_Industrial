from __future__ import annotations

import uuid
from pathlib import Path

import cv2
import numpy as np

from .config import DATA_DIR, MODEL_DIR, BoardTemplate, ensure_directories
from .database import add_cell, create_board
from .preprocessing import grid_quality, rectify, segment_fixed_grid
from .recognizer import LocalRecognizer
from .validation import validate_value


class ProcessingService:
    def __init__(self, template: BoardTemplate, recognizer: LocalRecognizer | None = None):
        self.template = template
        self.recognizer = recognizer or LocalRecognizer(MODEL_DIR)

    def process(self, filename: str, content: bytes, corners=None) -> dict:
        ensure_directories()
        image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Arquivo não é uma imagem válida")
        token = uuid.uuid4().hex
        raw_path = DATA_DIR / "raw_images" / f"{token}.jpg"
        raw_path.write_bytes(content)
        normalized = rectify(image, self.template, corners)
        board_id = create_board(filename, str(raw_path), self.template.name)
        segmented = list(segment_fixed_grid(normalized, self.template))
        candidates, candidate_indexes = [], []
        predictions = [("", 1.0) for _ in segmented]
        for index, (_, _, crop, _) in enumerate(segmented):
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            ink_ratio = float(np.count_nonzero(ink) / ink.size)
            if ink_ratio >= 0.012:
                candidates.append(crop); candidate_indexes.append(index)
        for index, prediction in zip(candidate_indexes, self.recognizer.predict_many(candidates)):
            predictions[index] = prediction
        results = []
        for index, (row, field, crop, box) in enumerate(segmented):
            cell_path = DATA_DIR / "cells" / f"{board_id}_{row}_{field.name}.png"
            cv2.imwrite(str(cell_path), crop)
            prediction, confidence = predictions[index]
            prediction, type_valid = validate_value(field, prediction)
            is_blank = index not in candidate_indexes
            needs_review = (not is_blank) and (confidence < self.template.confidence_threshold or not type_valid)
            cell_id = add_cell(board_id, row, field.name, str(cell_path), prediction, confidence, needs_review)
            results.append({"id": cell_id, "row": row, "field": field.name, "prediction": prediction, "confidence": confidence, "needs_review": needs_review, "box": box})
        return {"board_id": board_id, "grid_quality": grid_quality(normalized, self.template), "cells": results}
