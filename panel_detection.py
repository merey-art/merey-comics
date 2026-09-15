#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROW_Y_OVERLAP_THRESHOLD = 0.5


def load_model(weights_path: str) -> YOLO:
    return YOLO(weights_path)


def run_inference(model: YOLO, image_path: str, conf: float = 0.25, imgsz: int = 1280):
    results = model.predict(source=image_path, conf=conf, imgsz=imgsz, verbose=False)
    return results[0]


def extract_boxes(result) -> list[list[float]]:
    boxes = []
    if result.boxes is None:
        return boxes
    for xyxy in result.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        boxes.append([x1, y1, x2, y2])
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


def detect_panels(image_path: str, weights_path: str = "yolov8n.pt", conf: float = 0.25) -> dict:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]

    model = load_model(weights_path)
    result = run_inference(model, image_path, conf=conf)
    boxes = extract_boxes(result)
    ordered_boxes = sort_reading_order(boxes)

    return build_panel_json(ordered_boxes, (height, width))


def main():
    parser = argparse.ArgumentParser(description="Comic panel detection with YOLOv8")
    parser.add_argument("image", help="Path to comic page image")
    parser.add_argument("--weights", default="yolov8n.pt", help="Path to model weights (.pt or .onnx)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--output", default=None, help="Path to write JSON output")
    parser.add_argument("--frontend", action="store_true", help="Emit frontend-ready crop/CSS transform payload")
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"Error: image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    panel_json = detect_panels(args.image, weights_path=args.weights, conf=args.conf)
    output = build_frontend_response(panel_json) if args.frontend else panel_json

    output_str = json.dumps(output, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(output_str, encoding="utf-8")
    else:
        print(output_str)


if __name__ == "__main__":
    main()
