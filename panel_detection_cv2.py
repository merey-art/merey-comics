#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROW_Y_OVERLAP_THRESHOLD = 0.5
MIN_AREA_RATIO = 0.01
MAX_AREA_RATIO = 0.95


def preprocess(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Panels are separated by a near-white page background/gutter.
    # Isolate that background, then invert so each panel becomes one
    # solid foreground blob.
    _, bg_mask = cv2.threshold(blurred, 235, 255, cv2.THRESH_BINARY)
    panel_mask = cv2.bitwise_not(bg_mask)

    # Small kernel: patch tiny gaps (anti-aliased borders, thin line
    # breaks) without bridging the wider gutter between panels.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(panel_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closed


def find_panel_contours(mask: np.ndarray, image_shape: tuple[int, int]) -> list[list[float]]:
    height, width = image_shape
    page_area = float(height * width)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        area_ratio = area / page_area
        if area_ratio < MIN_AREA_RATIO or area_ratio > MAX_AREA_RATIO:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        rect_area = float(w * h)
        if rect_area == 0:
            continue

        # Keep only contours that are reasonably rectangular (a panel
        # border), discarding stray blobs/speech bubbles.
        extent = area / rect_area
        if extent < 0.6:
            continue

        aspect = w / h if h > 0 else 0
        if aspect < 0.15 or aspect > 8:
            continue

        boxes.append([float(x), float(y), float(x + w), float(y + h)])

    return boxes


def sort_reading_order(boxes: list[list[float]]) -> list[list[float]]:
    if not boxes:
        return []

    items = [{"bbox": b, "cy": (b[1] + b[3]) / 2, "h": b[3] - b[1]} for b in boxes]
    items.sort(key=lambda it: it["cy"])

    rows: list[list[dict]] = []
    for item in items:
        placed = False
        for row in rows:
            ref = row[0]
            overlap = min(item["h"], ref["h"]) * ROW_Y_OVERLAP_THRESHOLD
            if abs(item["cy"] - ref["cy"]) < overlap:
                row.append(item)
                placed = True
                break
        if not placed:
            rows.append([item])

    rows.sort(key=lambda row: min(it["bbox"][1] for it in row))

    ordered = []
    for row in rows:
        row.sort(key=lambda it: it["bbox"][0])
        ordered.extend(it["bbox"] for it in row)
    return ordered


def build_panel_json(boxes: list[list[float]], image_shape: tuple[int, int]) -> dict:
    height, width = image_shape
    panels = []
    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        panels.append({
            "index": idx,
            "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
        })
    return {"image_width": width, "image_height": height, "panels": panels}


def build_frontend_response(panel_json: dict) -> dict:
    width = panel_json["image_width"]
    height = panel_json["image_height"]
    panels = []
    for panel in panel_json["panels"]:
        x1, y1, x2, y2 = panel["bbox"]
        w = x2 - x1
        h = y2 - y1
        panels.append({
            "index": panel["index"],
            "bbox": panel["bbox"],
            "crop": {"x": x1, "y": y1, "width": w, "height": h},
            "style": {
                "left": f"{(x1 / width) * 100:.4f}%",
                "top": f"{(y1 / height) * 100:.4f}%",
                "width": f"{(w / width) * 100:.4f}%",
                "height": f"{(h / height) * 100:.4f}%",
                "transform": f"translate(-{x1}px, -{y1}px)",
            },
        })
    return {"image_width": width, "image_height": height, "panels": panels}


def detect_panels(image_path: str) -> dict:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]

    mask = preprocess(image)
    boxes = find_panel_contours(mask, (height, width))
    ordered_boxes = sort_reading_order(boxes)

    return build_panel_json(ordered_boxes, (height, width))


def main():
    parser = argparse.ArgumentParser(description="Comic panel detection with pure OpenCV")
    parser.add_argument("image", help="Path to comic page image")
    parser.add_argument("--output", default=None, help="Path to write JSON output")
    parser.add_argument("--frontend", action="store_true", help="Emit frontend-ready crop/CSS transform payload")
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"Error: image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    panel_json = detect_panels(args.image)
    output = build_frontend_response(panel_json) if args.frontend else panel_json

    output_str = json.dumps(output, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(output_str, encoding="utf-8")
    else:
        print(output_str)


if __name__ == "__main__":
    main()
