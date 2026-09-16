#!/usr/bin/env python3
"""Continue fine-tuning the panel detector on manually-labeled batches
(annotated in makesense.ai, YOLO .txt export).

Each source is a directory containing the page images directly and a
`labels/` subfolder with one same-named .txt per labeled image (pages
without a label file are skipped — not every page has to be annotated).
Combines all sources into one YOLO dataset, splits train/val, and
continues training from backend/models/comic_yolo.pt (falling back to
yolov8n.pt if that doesn't exist yet), overwriting comic_yolo.pt with the
result.
"""
import argparse
import random
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
YOLO_DATA_DIR = PROJECT_ROOT / "data" / "manual_finetune_yolo"
BASE_WEIGHTS = PROJECT_ROOT / "backend" / "models" / "comic_yolo.pt"
FALLBACK_BASE_WEIGHTS = PROJECT_ROOT / "yolov8n.pt"
RUNS_DIR = PROJECT_ROOT / "runs"
RUN_NAME = "manual_finetune"
OUTPUT_WEIGHTS = PROJECT_ROOT / "backend" / "models" / "comic_yolo.pt"
VAL_SPLIT = 0.15
SEED = 0
EPOCHS = 30
IMG_SIZE = 640
BATCH = 16
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def collect_pairs(source_dirs: list[Path]) -> list[tuple[Path, Path]]:
    """Return (image_path, label_path) pairs across all sources, skipping
    any page that has no matching label file."""
    pairs = []
    for src in source_dirs:
        labels_dir = src / "labels"
        if not labels_dir.is_dir():
            print(f"Warning: no labels/ subfolder in {src}, skipping it entirely.", file=sys.stderr)
            continue
        for label_path in sorted(labels_dir.glob("*.txt")):
            stem = label_path.stem
            image_path = None
            for ext in IMAGE_EXTS:
                candidate = src / f"{stem}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
            if image_path is None:
                print(f"Warning: no image found for label {label_path}, skipping.", file=sys.stderr)
                continue
            pairs.append((image_path, label_path))
    return pairs


def build_yolo_dataset(pairs: list[tuple[Path, Path]]) -> Path:
    if YOLO_DATA_DIR.exists():
        shutil.rmtree(YOLO_DATA_DIR)
    for split in ("train", "val"):
        (YOLO_DATA_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DATA_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    shuffled = pairs[:]
    random.Random(SEED).shuffle(shuffled)
    split_idx = max(1, int(len(shuffled) * (1 - VAL_SPLIT)))

    for i, (image_path, label_path) in enumerate(shuffled):
        split = "train" if i < split_idx else "val"
        # Flatten into unique filenames — two batches could otherwise
        # collide on generic page-number stems like "0004".
        unique_stem = f"{image_path.parent.name}_{image_path.stem}"
        unique_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in unique_stem)
        shutil.copy2(image_path, YOLO_DATA_DIR / "images" / split / f"{unique_stem}{image_path.suffix}")
        shutil.copy2(label_path, YOLO_DATA_DIR / "labels" / split / f"{unique_stem}.txt")

    n_train = split_idx
    n_val = len(shuffled) - split_idx
    print(f"Built dataset: {n_train} train / {n_val} val ({len(shuffled)} total labeled pages)")

    data_yaml = YOLO_DATA_DIR / "data.yaml"
    data_yaml.write_text(
        f"path: {YOLO_DATA_DIR}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: panel\n"
    )
    return data_yaml


def train(data_yaml: Path) -> Path:
    from ultralytics import YOLO

    base_weights = BASE_WEIGHTS if BASE_WEIGHTS.exists() else FALLBACK_BASE_WEIGHTS
    print(f"Continuing fine-tune from {base_weights}")

    model = YOLO(str(base_weights))
    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        project=str(RUNS_DIR),
        name=RUN_NAME,
        exist_ok=True,
    )

    best = RUNS_DIR / RUN_NAME / "weights" / "best.pt"
    if not best.exists():
        sys.exit(f"Training finished but {best} wasn't produced — check the run output above.")
    return best


def main():
    parser = argparse.ArgumentParser(description="Fine-tune the panel detector on manually-labeled batches")
    parser.add_argument(
        "sources",
        nargs="+",
        type=Path,
        help="Directories, each with page images + a labels/ subfolder (YOLO .txt export from makesense.ai)",
    )
    parser.add_argument("--keep-data", action="store_true", help="Don't delete the built dataset copy after training")
    args = parser.parse_args()

    for src in args.sources:
        if not src.is_dir():
            sys.exit(f"Not a directory: {src}")

    pairs = collect_pairs(args.sources)
    if not pairs:
        sys.exit("No labeled image/label pairs found across the given sources.")
    print(f"Found {len(pairs)} labeled pages across {len(args.sources)} source(s).")

    data_yaml = build_yolo_dataset(pairs)
    best_weights = train(data_yaml)

    OUTPUT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, OUTPUT_WEIGHTS)
    print(f"Saved fine-tuned weights to {OUTPUT_WEIGHTS}")

    if not args.keep_data:
        shutil.rmtree(YOLO_DATA_DIR, ignore_errors=True)
        print(f"Removed dataset working copy under {YOLO_DATA_DIR}")


if __name__ == "__main__":
    main()
