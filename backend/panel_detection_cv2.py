#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROW_Y_OVERLAP_THRESHOLD = 0.5
MIN_AREA_RATIO = 0.05
MAX_AREA_RATIO = 0.95
MIN_PANELS_FOR_FALLBACK = 2
MERGE_OVERLAP_RATIO = 0.3
GUTTER_EDGE_FRACTION = 0.004
MIN_STRIP_RATIO = 0.03
MIN_GUTTER_SPAN_RATIO = 0.002
MAX_REASONABLE_PANELS = 12
BUBBLE_MEAN_THRESHOLD = 225.0
BUBBLE_STD_THRESHOLD = 18.0
# Global mean/std over the whole box is fooled by a box that's almost
# entirely blank margin but grazes a thin border line from a neighboring
# panel (e.g. one row of a panel's bottom edge) — that line alone pushes
# std well past BUBBLE_STD_THRESHOLD even though the box has no real
# content. Edge *density* (fraction of Canny-edge pixels) isn't: a stray
# border line is a tiny fraction of the box's area, while real panel
# content (line art, shading, text) covers a large share of it.
BUBBLE_EDGE_DENSITY_THRESHOLD = 0.05
# A detector box that stops short of the page edge is either a real
# margin/border (blank strip between the art and the edge) or a
# full-bleed panel whose box the model just didn't extend all the way.
# Tiny gaps are snapped unconditionally (measurement noise); bigger ones
# are only snapped when the strip between the box and the edge actually
# has art in it, judged with the same blank/bubble heuristic used
# elsewhere (high mean, low std == blank).
EDGE_AUTO_SNAP_RATIO = 0.03


def preprocess(image: np.ndarray) -> np.ndarray:
    """Build a panel-content mask that works regardless of page background
    polarity (bright pages, black gutters, etc.).

    A fixed "background is near-white" threshold breaks completely on dark
    pages — the whole page reads as foreground. Detecting panel *content*
    via edges sidesteps that: line art, shading and text all produce strong
    Canny edges, while a gutter (black or white, as long as it's a flat
    color) produces almost none. Morphological closing then bridges gaps
    between disjoint edge segments into solid per-panel blobs.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]

    # Blend luminance with HSV value: mostly redundant, but value tracks
    # perceived brightness better on tinted (non-gray) panel backgrounds.
    blended = cv2.addWeighted(gray, 0.5, value, 0.5, 0)
    blurred = cv2.GaussianBlur(blended, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    # Bridge gaps in panel borders / linework into solid blocks. Sized
    # relative to the page's own resolution, not a fixed pixel count: a
    # high-res scan can have real gutters as thin as ~15px, and a kernel
    # tuned for a smaller test image bridges straight across them, fusing
    # separate panels into one blob (the opposite failure from too little
    # bridging on dark pages).
    min_dim = min(image.shape[:2])
    close_size = max(3, int(round(min_dim * 0.0008)) | 1)  # odd
    dilate_size = max(3, int(round(min_dim * 0.0012)) | 1)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    # Thicken further so a panel's interior edges merge into one filled
    # blob instead of a hollow outline.
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_size, dilate_size))
    mask = cv2.dilate(closed, dilate_kernel, iterations=1)

    return mask


def _is_blank_or_bubble(image: np.ndarray, box: list[float]) -> bool:
    """Reject a region that's essentially a flat color patch or a small
    speech bubble with little text in it, rather than real panel content."""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    region = image[max(y1, 0):y2, max(x1, 0):x2]
    if region.size == 0:
        return True
    gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 else region
    mean, std = float(gray_region.mean()), float(gray_region.std())
    if mean <= BUBBLE_MEAN_THRESHOLD:
        return False
    if std < BUBBLE_STD_THRESHOLD:
        return True
    edge_density = float((cv2.Canny(gray_region, 50, 150) > 0).mean())
    return edge_density < BUBBLE_EDGE_DENSITY_THRESHOLD


def find_panel_contours(
    mask: np.ndarray, image_shape: tuple[int, int], source_image: np.ndarray | None = None
) -> list[list[float]]:
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

        aspect = w / h if h > 0 else 0
        if aspect < 0.15 or aspect > 8:
            continue

        box = [float(x), float(y), float(x + w), float(y + h)]
        if source_image is not None and _is_blank_or_bubble(source_image, box):
            continue

        boxes.append(box)

    return boxes


def _projection_gutter_bands(is_gutter: np.ndarray, min_span: int) -> list[tuple[int, int]]:
    """Collapse a per-row/column boolean "is this a gutter" profile into
    contiguous bands, dropping ones thinner than `min_span`.

    Bands thinner than `min_span` are dropped rather than treated as real
    separators: a single stray gutter-like row/column is as likely to be
    noise (a sparse patch of flat art) as an actual gutter, and splitting
    on it shreds busy full-bleed pages into bogus fragments.
    """
    bands = []
    start = None
    for i, flag in enumerate(is_gutter):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            bands.append((start, i))
            start = None
    if start is not None:
        bands.append((start, len(is_gutter)))
    return [(s, e) for s, e in bands if (e - s) >= min_span]


def _segments_between_bands(bands: list[tuple[int, int]], total: int) -> list[tuple[int, int]]:
    segments = []
    cursor = 0
    for start, end in bands:
        if start > cursor:
            segments.append((cursor, start))
        cursor = end
    if cursor < total:
        segments.append((cursor, total))
    return segments


def split_by_projection(gray: np.ndarray) -> list[list[float]]:
    """Recursive X-Y cut: split the page into horizontal strips at solid
    row-wise separators, then split each strip into panels at solid
    column-wise separators.

    A gutter row/column is one with almost no Canny edges — line art,
    shading and text all produce edges, a flat-colored separator doesn't,
    regardless of whether it's black or white. Edge *density* is used
    rather than raw intensity variance: a two-page spread or a page with
    a faint full-height accent line keeps every row's brightness variance
    elevated even where there's a genuine gutter, which made the old
    variance-based profile blind to real separators on those pages.
    """
    height, width = gray.shape[:2]
    edges = cv2.Canny(gray, 50, 150)
    edge_frac = edges.astype(np.float32) / 255.0

    min_row_gutter = max(4, int(height * MIN_GUTTER_SPAN_RATIO))
    min_col_gutter = max(4, int(width * MIN_GUTTER_SPAN_RATIO))

    row_is_gutter = edge_frac.mean(axis=1) < GUTTER_EDGE_FRACTION
    row_bands = _projection_gutter_bands(row_is_gutter, min_row_gutter)
    strips = _segments_between_bands(row_bands, height)

    boxes = []
    for y1, y2 in strips:
        if (y2 - y1) < height * MIN_STRIP_RATIO:
            continue
        strip_edges = edge_frac[y1:y2, :]
        col_is_gutter = strip_edges.mean(axis=0) < GUTTER_EDGE_FRACTION
        col_bands = _projection_gutter_bands(col_is_gutter, min_col_gutter)
        segments = _segments_between_bands(col_bands, width)

        for x1, x2 in segments:
            if (x2 - x1) < width * MIN_STRIP_RATIO:
                continue
            boxes.append([float(x1), float(y1), float(x2), float(y2)])

    # A real comic page rarely has more than a dozen panels; a larger
    # count means the profile picked up noise (sparse art leaving stray
    # flat bands) rather than genuine gutters, so don't trust it.
    if len(boxes) > MAX_REASONABLE_PANELS:
        return []

    return boxes


def _box_area(box: list[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _intersection_area(a: list[float], b: list[float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _union_box(a: list[float], b: list[float]) -> list[float]:
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def merge_overlapping_boxes(
    boxes: list[list[float]], overlap_ratio: float = MERGE_OVERLAP_RATIO
) -> list[list[float]]:
    """Collapse duplicate/nested contours into a single box per panel.

    Two boxes are merged when their intersection covers a large share of the
    smaller one (nested inset panels, double-detected borders), not merely
    when they touch — real neighboring panels only share a thin gutter edge
    and shouldn't be combined.
    """
    merged = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                a, b = merged[i], merged[j]
                inter = _intersection_area(a, b)
                if inter == 0:
                    continue
                smaller_area = min(_box_area(a), _box_area(b))
                if smaller_area == 0:
                    continue
                if inter / smaller_area >= overlap_ratio:
                    merged[i] = _union_box(a, b)
                    del merged[j]
                    changed = True
                    break
            if changed:
                break
    return merged


def _rects_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return max(ax1, bx1) < min(ax2, bx2) and max(ay1, by1) < min(ay2, by2)


def _has_neighbor_in_gap(
    boxes: list[list[float]], index: int, gap: tuple[float, float, float, float]
) -> bool:
    """True if some other box overlaps the rectangular gap between this
    box and the page edge — i.e. the gap isn't empty page margin, another
    panel is (at least partly) sitting in it. `boxes` must be the
    original (unmutated) detections — reusing a list whose earlier
    entries were already snapped this pass would shift where their
    boundaries actually are."""
    for j, other in enumerate(boxes):
        if j == index:
            continue
        if _rects_overlap(tuple(other), gap):
            return True
    return False


def _strip_is_blank(image: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> bool:
    xi1, yi1, xi2, yi2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    if xi2 <= xi1 or yi2 <= yi1:
        return True
    strip = image[yi1:yi2, xi1:xi2]
    if strip.size == 0:
        return True
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY) if strip.ndim == 3 else strip
    return gray.mean() > BUBBLE_MEAN_THRESHOLD and gray.std() < BUBBLE_STD_THRESHOLD


def snap_full_bleed_edges(
    boxes: list[list[float]], image: np.ndarray
) -> list[list[float]]:
    """Extend a box's edge to the page boundary when the gap is either
    tiny (model measurement noise) or clearly full-bleed artwork rather
    than a blank margin — and only when nothing else (a neighboring
    panel) already occupies that gap."""
    if not boxes:
        return boxes

    height, width = image.shape[:2]
    original = [tuple(b) for b in boxes]
    snapped = [list(b) for b in boxes]

    for i, box in enumerate(snapped):
        x1, y1, x2, y2 = original[i]

        gap = (0.0, y1, x1, y2)
        if x1 > 0 and not _has_neighbor_in_gap(original, i, gap):
            if x1 / width <= EDGE_AUTO_SNAP_RATIO or not _strip_is_blank(image, *gap):
                box[0] = 0.0

        gap = (x1, 0.0, x2, y1)
        if y1 > 0 and not _has_neighbor_in_gap(original, i, gap):
            if y1 / height <= EDGE_AUTO_SNAP_RATIO or not _strip_is_blank(image, *gap):
                box[1] = 0.0

        gap = (x2, y1, float(width), y2)
        if x2 < width and not _has_neighbor_in_gap(original, i, gap):
            if (width - x2) / width <= EDGE_AUTO_SNAP_RATIO or not _strip_is_blank(image, *gap):
                box[2] = float(width)

        gap = (x1, y2, x2, float(height))
        if y2 < height and not _has_neighbor_in_gap(original, i, gap):
            if (height - y2) / height <= EDGE_AUTO_SNAP_RATIO or not _strip_is_blank(image, *gap):
                box[3] = float(height)

    return snapped


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


def detect_panel_boxes(image: np.ndarray) -> list[list[float]]:
    """Run the full detection pipeline and return ordered panel boxes,
    falling back to a single full-page box when detection is unreliable
    (no panels, or too few for a page that isn't a genuine splash page)."""
    height, width = image.shape[:2]

    mask = preprocess(image)
    boxes = find_panel_contours(mask, (height, width), source_image=image)
    boxes = merge_overlapping_boxes(boxes)

    # The closing/dilation step that bridges broken linework into solid
    # blobs can, on a high-res page with thin gutters, bridge straight
    # across a real gutter too and fuse separate panels into one. The
    # projection cut is immune to that (it works off intensity variance,
    # not edge geometry) so it's run as a second opinion whenever it might
    # catch a split the contour pass missed — not just when the contour
    # pass finds nothing.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    projected = split_by_projection(gray)
    projected = [b for b in projected if not _is_blank_or_bubble(image, b)]
    projected = merge_overlapping_boxes(projected)

    if len(projected) > len(boxes):
        boxes = projected

    if len(boxes) < MIN_PANELS_FOR_FALLBACK:
        return [[0.0, 0.0, float(width), float(height)]]

    boxes = snap_full_bleed_edges(boxes, image)
    return sort_reading_order(boxes)


def detect_panels(image_path: str) -> dict:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]

    ordered_boxes = detect_panel_boxes(image)

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
