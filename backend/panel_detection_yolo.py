#!/usr/bin/env python3
"""YOLO-based comic panel detection.

Prefers our own fine-tuned model (backend/models/comic_yolo.pt — see
../train_panel_detector.py) when it exists, otherwise falls back to
mosesb/best-comic-panel-detection (YOLOv12x, single class "Comic Panel",
mAP50 0.991 on its own validation set). Either way, falls back further to
the Canny/morphology CV pipeline when no model is available or nothing is
found, and to a full-page box as the last resort — see detect_panels().
"""
import urllib.request
from pathlib import Path

import numpy as np

from panel_detection_cv2 import (
    MIN_AREA_RATIO,
    _box_area,
    _intersection_area,
    _is_blank_or_bubble,
    detect_panel_boxes,
    snap_full_bleed_edges,
    sort_reading_order,
)

MODELS_DIR = Path(__file__).resolve().parent / "models"
FINETUNED_WEIGHTS = MODELS_DIR / "comic_yolo.pt"
PRETRAINED_WEIGHTS = MODELS_DIR / "comic_panel_yolov12x.pt"
MODEL_URL = "https://huggingface.co/mosesb/best-comic-panel-detection/resolve/main/best.pt"
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.35
IMG_SIZE = 1280
# A real panel doesn't come in as a razor-thin sliver — the model
# sometimes fires on the diagonal/angled *gutter* between two panels
# (which is exactly that shape) and calls it a panel of its own. Reusing
# the CV pipeline's area floor plus a minimum-side-length check screens
# those out without touching genuinely tall/wide panels.
MIN_BOX_SIDE_RATIO = 0.06
# IoU-based NMS (via the `iou` argument above) only suppresses boxes that
# overlap heavily as a *fraction of their union*. A near-duplicate that's
# offset (e.g. one box drawn a bit wider than the other for the same
# panel) can have a low IoU while still having one box almost entirely
# *contained* inside the other — that's the shape a duplicate takes when
# the model is unsure about one edge. Catch those by containment instead.
CONTAINMENT_RATIO = 0.8

_model = None
_model_load_failed = False


def _resolve_weights_path() -> Path:
    return FINETUNED_WEIGHTS if FINETUNED_WEIGHTS.exists() else PRETRAINED_WEIGHTS


def _ensure_weights_downloaded(weights_path: Path) -> bool:
    """The pretrained weight file (~120MB) isn't committed to the repo
    (see .gitignore), so a fresh checkout fetches it on first use instead
    of requiring a manual download step. Our own fine-tune has nowhere to
    download from — it only exists once train_panel_detector.py has
    actually been run."""
    if weights_path.exists():
        return True
    if weights_path != PRETRAINED_WEIGHTS:
        return False
    try:
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = weights_path.with_suffix(".pt.download")
        urllib.request.urlretrieve(MODEL_URL, tmp_path)
        tmp_path.rename(weights_path)
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

        weights_path = _resolve_weights_path()
        if not _ensure_weights_downloaded(weights_path):
            _model_load_failed = True
            return None
        _model = YOLO(str(weights_path))
        return _model
    except Exception:
        # Missing/broken ultralytics install, corrupt weights, etc. — the
        # caller falls back to the CV pipeline rather than failing the
        # request outright.
        _model_load_failed = True
        return None


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


def _drop_blank_boxes(
    detections: list[tuple[list[float], float]], image: np.ndarray
) -> list[tuple[list[float], float]]:
    """Reject boxes that are just flat/blank page margin — the model
    sometimes fires on empty space (e.g. below the last row of panels)
    since it isn't trained to abstain there the way the CV contour pass
    does via _is_blank_or_bubble()."""
    return [(box, score) for box, score in detections if not _is_blank_or_bubble(image, box)]


def _drop_contained_duplicates(
    detections: list[tuple[list[float], float]]
) -> list[tuple[list[float], float]]:
    """Collapse near-duplicate detections of the same panel that IoU-based
    NMS misses — e.g. one box drawn a bit wider than the other, so most of
    the smaller box's area sits inside the larger one despite their IoU
    (overlap over *union*) staying below the NMS threshold. Keeps
    whichever of the pair has the higher confidence."""
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
    image: np.ndarray, conf: float = CONF_THRESHOLD, iou: float = IOU_THRESHOLD
) -> list[list[float]]:
    """Run YOLO panel detection and return boxes in reading order.

    Non-max suppression (duplicate/heavily-overlapping box removal) is
    handled by Ultralytics internally during predict() via the `iou`
    argument; _drop_contained_duplicates() catches near-duplicates that
    slip past IoU-based NMS (see its docstring).
    """
    model = _get_model()
    if model is None:
        return []

    results = model.predict(source=image, conf=conf, iou=iou, imgsz=IMG_SIZE, verbose=False)
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []

    detections = list(zip(
        [[float(v) for v in xyxy] for xyxy in result.boxes.xyxy.cpu().numpy()],
        [float(c) for c in result.boxes.conf.cpu().numpy()],
    ))
    detections = _drop_sliver_boxes(detections, image.shape[:2])
    detections = _drop_blank_boxes(detections, image)
    detections = _drop_contained_duplicates(detections)
    boxes = [box for box, _score in detections]
    boxes = snap_full_bleed_edges(boxes, image)
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
