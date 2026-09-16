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

from panel_detection_cv2 import (
    MIN_AREA_RATIO,
    _box_area,
    _intersection_area,
    build_frontend_response,
    build_panel_json,
    detect_panel_boxes,
    sort_reading_order,
)

MODEL_URL = "https://huggingface.co/mosesb/best-comic-panel-detection/resolve/main/best.pt"
MODELS_DIR = Path(__file__).resolve().parent / "backend" / "models"
# Our own fine-tune (see train_panel_detector.py) beats the generic
# pretrained model whenever it's been trained, but isn't there until
# someone runs that script.
FINETUNED_WEIGHTS = MODELS_DIR / "comic_yolo.pt"
PRETRAINED_WEIGHTS = MODELS_DIR / "comic_panel_yolov12x.pt"
DEFAULT_WEIGHTS = FINETUNED_WEIGHTS if FINETUNED_WEIGHTS.exists() else PRETRAINED_WEIGHTS
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.35
IMG_SIZE = 1280
# A real panel isn't a razor-thin sliver — the model sometimes fires on
# the angled gutter strip between two panels (exactly that shape) and
# calls it a panel of its own. Reusing the CV pipeline's area floor plus
# a minimum-side-length check screens those out.
MIN_BOX_SIDE_RATIO = 0.06
# IoU-based NMS (the `iou` argument above) only suppresses boxes that
# overlap heavily as a fraction of their *union*. A near-duplicate that's
# offset (one box drawn a bit wider than the other for the same panel)
# can have a low IoU while one box still sits almost entirely *inside*
# the other — catch those by containment instead, keeping whichever has
# the higher confidence.
CONTAINMENT_RATIO = 0.8


def ensure_weights(weights_path: Path) -> bool:
    if weights_path.exists():
        return True
    if weights_path != PRETRAINED_WEIGHTS:
        # Anything other than the known pretrained-model path (e.g. our
        # own fine-tune) has to come from actually running
        # train_panel_detector.py — there's nowhere to auto-download it
        # from.
        return False
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


def _drop_sliver_boxes(
    detections: list[tuple[list[float], float]], image_shape: tuple[int, int]
) -> list[tuple[list[float], float]]:
    """Reject boxes too small or too thin to be a real panel — typically
    the model firing on the angled gutter strip between two panels rather
    than a panel itself."""
    height, width = image_shape
    page_area = float(height * width)
    kept = []
    for box, score in detections:
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        if (w * h) / page_area < MIN_AREA_RATIO:
            continue
        if w / width < MIN_BOX_SIDE_RATIO or h / height < MIN_BOX_SIDE_RATIO:
            continue
        kept.append((box, score))
    return kept


def _drop_contained_duplicates(
    detections: list[tuple[list[float], float]]
) -> list[tuple[list[float], float]]:
    """Collapse near-duplicate detections of the same panel that IoU-based
    NMS misses (see CONTAINMENT_RATIO above). Keeps whichever of a
    containment pair has the higher confidence."""
    kept = list(detections)
    changed = True
    while changed:
        changed = False
        for i in range(len(kept)):
            for j in range(i + 1, len(kept)):
                box_a, score_a = kept[i]
                box_b, score_b = kept[j]
                inter = _intersection_area(box_a, box_b)
                area_a, area_b = _box_area(box_a), _box_area(box_b)
                if area_a == 0 or area_b == 0:
                    continue
                if inter / area_a >= CONTAINMENT_RATIO or inter / area_b >= CONTAINMENT_RATIO:
                    drop_j = score_a >= score_b
                    del kept[j if drop_j else i]
                    changed = True
                    break
            if changed:
                break
    return kept


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

    detections = list(zip(
        [[float(v) for v in xyxy] for xyxy in result.boxes.xyxy.cpu().numpy()],
        [float(c) for c in result.boxes.conf.cpu().numpy()],
    ))
    detections = _drop_sliver_boxes(detections, image.shape[:2])
    detections = _drop_contained_duplicates(detections)
    boxes = [box for box, _score in detections]
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
