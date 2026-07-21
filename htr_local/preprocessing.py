from __future__ import annotations

import cv2
import numpy as np

from .config import BoardTemplate


def order_corners(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    result = np.zeros((4, 2), dtype=np.float32)
    sums, diffs = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    result[0], result[2] = pts[np.argmin(sums)], pts[np.argmax(sums)]
    result[1], result[3] = pts[np.argmin(diffs)], pts[np.argmax(diffs)]
    return result


def detect_board_corners(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4 and cv2.contourArea(approx) > image.shape[0] * image.shape[1] * 0.25:
            return order_corners(approx)
    h, w = image.shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)


def rectify(image: np.ndarray, template: BoardTemplate, corners: np.ndarray | None = None) -> np.ndarray:
    source = order_corners(corners) if corners is not None else detect_board_corners(image)
    target = np.array([[0, 0], [template.width - 1, 0], [template.width - 1, template.height - 1], [0, template.height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(source, target)
    warped = cv2.warpPerspective(image, matrix, (template.width, template.height))
    lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
    light, a, b = cv2.split(lab)
    light = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(light)
    return cv2.cvtColor(cv2.merge((light, a, b)), cv2.COLOR_LAB2BGR)


def segment_fixed_grid(image: np.ndarray, template: BoardTemplate):
    gx1, gy1, gx2, gy2 = template.grid_bounds
    left, top = int(gx1 * template.width), int(gy1 * template.height)
    right, bottom = int(gx2 * template.width), int(gy2 * template.height)
    row_height = (bottom - top) / template.rows
    bounds = template.column_bounds or tuple(i / len(template.fields) for i in range(len(template.fields) + 1))
    for row in range(template.rows):
        y1, y2 = int(top + row * row_height), int(top + (row + 1) * row_height)
        for col, field in enumerate(template.fields):
            x1, x2 = int(left + bounds[col] * (right - left)), int(left + bounds[col + 1] * (right - left))
            m = template.crop_margin
            yield row, field, image[y1 + m:y2 - m, x1 + m:x2 - m].copy(), (x1, y1, x2, y2)


def grid_quality(image: np.ndarray, template: BoardTemplate) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, template.width // 20), 1)))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, template.height // 20))))
    return min(1.0, float((np.count_nonzero(horizontal) + np.count_nonzero(vertical)) / (image.shape[0] * image.shape[1] * 0.08)))
