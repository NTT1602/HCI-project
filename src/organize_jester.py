"""Script to organize Jester dataset for Slide Presentation Controller HCI project.

Filters relevant gestures and organizes them into standard deep learning structure:
    D:\\Data HCI\\slide_gestures\\
        ├── train\\
        │   ├── Swiping Left\\
        │   ├── Swiping Right\\
        │   └── ...
        ├── validation\\
        │   └── ...
        └── test\\
            └── ...
"""
import os
import csv
import time
from collections import Counter
from pathlib import Path

# Paths
DATA_DIR = r"D:\Data HCI\20bn-jester-v1"
LABELS_DIR = r"D:\Data HCI\jester-labels\jester_labels"
DEST_ROOT = r"D:\Data HCI\slide_gestures"

# Target gestures for Slide Presentation Controller
TARGET_LABELS = {
    "Swiping Left",
    "Swiping Right",
    "Stop Sign",
    "No gesture",
    "Doing other things",
    "Thumb Up",
    "Thumb Down",
    "Zooming In With Full Hand",
    "Zooming Out With Full Hand",
    "Zooming In With Two Fingers",
    "Zooming Out With Two Fingers",
}


def load_csv(labels_dir: str, filename: str) -> dict[str, str]:
    mapping = {}
    path = os.path.join(labels_dir, filename)
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.reader(f, delimiter=";"):
            if len(r) >= 2:
                mapping[r[0].strip()] = r[1].strip()
    return mapping


def main():
    print("=" * 60)
    print("Starting Jester Dataset Organization for Slide HCI Project")
    print(f"Source: {DATA_DIR}")
    print(f"Destination: {DEST_ROOT}")
    print("=" * 60)

    t0 = time.time()
    existing_dirs = set(os.listdir(DATA_DIR))
    print(f"Found {len(existing_dirs)} video folders in source directory.")

    splits = {
        "train": load_csv(LABELS_DIR, "train.csv"),
        "validation": load_csv(LABELS_DIR, "validation.csv"),
        "test": load_csv(LABELS_DIR, "test-answers.csv"),
    }

    manifest_rows = []
    stats = {split: Counter() for split in splits}
    moved_count = 0
    errors = []

    for split_name, mapping in splits.items():
        print(f"\nProcessing split: {split_name}...")
        for vid, label in mapping.items():
            if label not in TARGET_LABELS:
                continue
            if vid not in existing_dirs:
                continue

            src_folder = os.path.join(DATA_DIR, vid)
            dst_folder = os.path.join(DEST_ROOT, split_name, label, vid)

            try:
                os.makedirs(os.path.dirname(dst_folder), exist_ok=True)
                os.replace(src_folder, dst_folder)
                stats[split_name][label] += 1
                moved_count += 1
                manifest_rows.append([vid, split_name, label, dst_folder])

                if moved_count % 10000 == 0:
                    print(f"  Moved {moved_count} folders so far ({time.time() - t0:.1f}s)...")

            except Exception as e:
                errors.append((vid, str(e)))

    # Write manifest CSV
    manifest_path = os.path.join(DEST_ROOT, "manifest.csv")
    print(f"\nWriting manifest to {manifest_path}...")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "split", "label", "folder_path"])
        writer.writerows(manifest_rows)

    elapsed = time.time() - t0
    print("=" * 60)
    print(f"SUCCESS: Moved {moved_count} video folders in {elapsed:.2f} seconds.")
    if errors:
        print(f"Encountered {len(errors)} errors:")
        for err in errors[:5]:
            print(f"  {err}")
    print("=" * 60)

    # Print summary breakdown
    for split_name, counter in stats.items():
        print(f"\n--- Split: {split_name} (Total: {sum(counter.values())}) ---")
        for lbl, cnt in sorted(counter.items()):
            print(f"  {lbl:<30}: {cnt}")

    remaining = len(os.listdir(DATA_DIR))
    print(f"\nRemaining folders in original directory ({DATA_DIR}): {remaining}")


if __name__ == "__main__":
    main()
