#!/usr/bin/env python3
"""YOLO-based comic panel detection.

Uses a YOLO model fine-tuned specifically to detect comic panels
(mosesb/best-comic-panel-detection, YOLOv12x, single class "Comic Panel",
mAP50 0.991 on its validation set) instead of the generic Canny/morphology
heuristics in panel_detection_cv2.py. Falls back to that CV pipeline when
the model is unavailable or finds nothing, and to a full-page box as the
last resort — see detect_panels().
"""
import urllib.request
from pathlib import Path

import numpy as np

from panel_detection_cv2 import detect_panel_boxes, sort_reading_order

MODEL_PATH = Path(__file__).resolve().parent / "models" / "comic_panel_yolov12x.pt"
MODEL_URL = "https://huggingface.co/mosesb/best-comic-panel-detection/resolve/main/best.pt"
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.5
IMG_SIZE = 1280

_model = None
_model_load_failed = False


def _ensure_weights_downloaded() -> bool:
    """The weight file (~120MB) isn't committed to the repo (see
    .gitignore), so a fresh checkout fetches it on first use instead of
    requiring a manual download step."""
    if MODEL_PATH.exists():
        return True
    try:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = MODEL_PATH.with_suffix(".pt.download")
        urllib.request.urlretrieve(MODEL_URL, tmp_path)
        tmp_path.rename(MODEL_PATH)
        return True
    except Exception:
        return False


def _get_model():
    """Lazily load and cache the YOLO model. Import + weight loading is
    slow (seconds), so this must happen once per process, not per request."""
    global _model, _model_load_failed
    if _model is not None:
        return _model
    if _model_load_failed:
        return None
    try:
        from ultralytics import YOLO

        if not _ensure_weights_downloaded():
            _model_load_failed = True
            return None
        _model = YOLO(str(MODEL_PATH))
        return _model
    except Exception:
        # Missing/broken ultralytics install, corrupt weights, etc. — the
        # caller falls back to the CV pipeline rather than failing the
        # request outright.
        _model_load_failed = True
        return None


def detect_panels_yolo(
    image: np.ndarray, conf: float = CONF_THRESHOLD, iou: float = IOU_THRESHOLD
) -> list[list[float]]:
    """Run YOLO panel detection and return boxes in reading order.

    Non-max suppression (duplicate/heavily-overlapping box removal) is
    handled by Ultralytics internally during predict() via the `iou`
    argument, rather than reimplemented here.
    """
    model = _get_model()
    if model is None:
        return []

    results = model.predict(source=image, conf=conf, iou=iou, imgsz=IMG_SIZE, verbose=False)
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = [[float(v) for v in xyxy] for xyxy in result.boxes.xyxy.cpu().numpy()]
    return sort_reading_order(boxes)


def detect_panels(image: np.ndarray) -> list[list[float]]:
    """Full detection pipeline: YOLO first, then the CV heuristic
    pipeline, then a single full-page box — each stage only runs if the
    previous one found nothing."""
    boxes = detect_panels_yolo(image)
    if boxes:
        return boxes

    # detect_panel_boxes already falls all the way back to a full-page box
    # itself when its own contour/projection passes come up empty, so this
    # one call covers both the "CV" and "full page" fallback tiers.
    return detect_panel_boxes(image)
