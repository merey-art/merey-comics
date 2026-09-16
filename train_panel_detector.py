#!/usr/bin/env python3
"""Fine-tune YOLOv8 for comic panel detection.

Downloads a public, key-free comic panel dataset from Hugging Face
(kwanwoo02/Panel_Segmentation2speech_Balloon_Detection — 1545 pages,
5842 panel boxes, COCO format), converts it to YOLO format, and
fine-tunes yolov8n.pt on it. The resulting weights are copied to
backend/models/comic_yolo.pt, which panel_detection.py picks up ahead of
the generic pretrained model. Meant to be re-run whenever you want to
retrain on more/different data — it re-downloads and re-converts the
dataset each time rather than assuming a previous run's output is still
around.
"""
import argparse
import json
import random
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

DATASET_REPO = "kwanwoo02/Panel_Segmentation2speech_Balloon_Detection"
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "comic_panels_raw"
YOLO_DATA_DIR = PROJECT_ROOT / "data" / "comic_panels_yolo"
BASE_WEIGHTS = PROJECT_ROOT / "yolov8n.pt"
RUNS_DIR = PROJECT_ROOT / "runs"
RUN_NAME = "comic_panel_finetune"
OUTPUT_WEIGHTS = PROJECT_ROOT / "backend" / "models" / "comic_yolo.pt"
VAL_SPLIT = 0.1
SEED = 0
EPOCHS = 30
IMG_SIZE = 640
BATCH = 16


def download_dataset() -> Path:
    """Public dataset, public repo — no HF token / login required."""
    print(f"Downloading dataset {DATASET_REPO} from Hugging Face (public, no API key)...")
    local_dir = snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        allow_patterns=["panel/*", "annotations/panel_coco.json"],
        local_dir=RAW_DATA_DIR,
    )
    return Path(local_dir)


def coco_to_yolo(dataset_dir: Path) -> Path:
    """Convert the COCO-format panel annotations into a YOLOv8 dataset
    layout (images/{train,val}, labels/{train,val}) and write data.yaml."""
    coco_path = dataset_dir / "annotations" / "panel_coco.json"
    coco = json.loads(coco_path.read_text())

    images_by_id = {img["id"]: img for img in coco["images"]}
    anns_by_image: dict[int, list] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    image_ids = list(images_by_id.keys())
    random.Random(SEED).shuffle(image_ids)
    split_idx = max(1, int(len(image_ids) * (1 - VAL_SPLIT)))
    train_ids = set(image_ids[:split_idx])

    if YOLO_DATA_DIR.exists():
        shutil.rmtree(YOLO_DATA_DIR)
    for split in ("train", "val"):
        (YOLO_DATA_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DATA_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    skipped = 0
    written = 0
    for image_id, img in images_by_id.items():
        split = "train" if image_id in train_ids else "val"
        src = dataset_dir / "panel" / img["file_name"]
        if not src.exists():
            skipped += 1
            continue

        dst_img = YOLO_DATA_DIR / "images" / split / img["file_name"]
        shutil.copy2(src, dst_img)

        width, height = img["width"], img["height"]
        lines = []
        for ann in anns_by_image.get(image_id, []):
            x, y, w, h = ann["bbox"]
            cx = (x + w / 2) / width
            cy = (y + h / 2) / height
            nw = w / width
            nh = h / height
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        label_path = YOLO_DATA_DIR / "labels" / split / (Path(img["file_name"]).stem + ".txt")
        label_path.write_text("\n".join(lines))
        written += 1

    if skipped:
        print(f"Warning: {skipped} images referenced in annotations were missing on disk, skipped.")
    print(f"Converted {written} images ({len(train_ids)} train / {written - len(train_ids & set(images_by_id))} approx val).")

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

    if not BASE_WEIGHTS.exists():
        sys.exit(f"Base weights not found at {BASE_WEIGHTS} — expected yolov8n.pt in the project root.")

    model = YOLO(str(BASE_WEIGHTS))
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


def cleanup_dataset():
    """Delete the downloaded/converted dataset (several GB of page scans
    we only needed to produce best.pt) — re-running this script
    re-downloads it, so nothing is lost by removing it."""
    for path in (RAW_DATA_DIR, YOLO_DATA_DIR):
        if path.exists():
            shutil.rmtree(path)
    print(f"Removed dataset working copies under {RAW_DATA_DIR.parent}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 for comic panel detection")
    parser.add_argument(
        "--keep-data",
        action="store_true",
        help="Don't delete the downloaded/converted dataset (data/) after training. "
        "It's several GB; by default it's removed once best.pt is saved.",
    )
    args = parser.parse_args()

    dataset_dir = download_dataset()
    data_yaml = coco_to_yolo(dataset_dir)
    best_weights = train(data_yaml)

    OUTPUT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, OUTPUT_WEIGHTS)
    print(f"Saved fine-tuned weights to {OUTPUT_WEIGHTS}")
    print("panel_detection.py will use this automatically — see its DEFAULT_WEIGHTS resolution.")

    if args.keep_data:
        print(f"Keeping dataset files in {RAW_DATA_DIR.parent} (--keep-data passed).")
    else:
        cleanup_dataset()


if __name__ == "__main__":
    main()
