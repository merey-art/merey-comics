#!/usr/bin/env python3
"""Comic panel detection CLI.

Detects panels with a YOLO model fine-tuned for comic pages
(mosesb/best-comic-panel-detection), falling back to the Canny/morphology
CV heuristic (panel_detection_cv2.py) when the model is unavailable or
finds nothing, and to a single full-page box as the last resort.
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

import cv2

from panel_detection_cv2 import detect_panel_boxes, sort_reading_order, build_panel_json, build_frontend_response

MODEL_URL = "https://huggingface.co/mosesb/best-comic-panel-detection/resolve/main/best.pt"
DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "backend" / "models" / "comic_panel_yolov12x.pt"
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.5
IMG_SIZE = 1280


def ensure_weights(weights_path: Path) -> bool:
    if weights_path.exists():
        return True
    try:
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = weights_path.with_suffix(".pt.download")
        print(f"Downloading panel-detection weights to {weights_path}...", file=sys.stderr)
        urllib.request.urlretrieve(MODEL_URL, tmp_path)
        tmp_path.rename(weights_path)
        return True
    except Exception as exc:
        print(f"Warning: could not download weights ({exc})", file=sys.stderr)
        return False


def detect_panels_yolo(
    image, weights_path: Path, conf: float = CONF_THRESHOLD, iou: float = IOU_THRESHOLD
) -> list[list[float]]:
    if not ensure_weights(weights_path):
        return []
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Warning: ultralytics not installed, falling back to CV detection", file=sys.stderr)
        return []

    model = YOLO(str(weights_path))
    results = model.predict(source=image, conf=conf, iou=iou, imgsz=IMG_SIZE, verbose=False)
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = [[float(v) for v in xyxy] for xyxy in result.boxes.xyxy.cpu().numpy()]
    return sort_reading_order(boxes)


def detect_panels(image_path: str, weights_path: str = str(DEFAULT_WEIGHTS), conf: float = CONF_THRESHOLD) -> dict:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]

    ordered_boxes = detect_panels_yolo(image, Path(weights_path), conf=conf)
    if not ordered_boxes:
        # Covers both the CV fallback and, inside it, the full-page fallback.
        ordered_boxes = detect_panel_boxes(image)

    return build_panel_json(ordered_boxes, (height, width))


def main():
    parser = argparse.ArgumentParser(description="Comic panel detection (YOLO, with CV fallback)")
    parser.add_argument("image", help="Path to comic page image")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Path to YOLO model weights (.pt)")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="Confidence threshold")
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
