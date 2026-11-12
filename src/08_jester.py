"""Step 8: Extract 21 hand landmarks + motion vectors from Jester dataset.

Option 2 Implementation:
Combines 63 static landmarks (at keyframe) with dynamic motion vectors (dx, dy, dz)
between gesture start and end frames to effectively distinguish directional gestures
(such as Swiping Left vs. Swiping Right).

Optimized with multi-threading and auto-resume capability.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from landmark_utils import FEATURE_NAMES, ensure_parent

# Dynamic motion feature names
MOTION_FEATURE_NAMES = [
    "dx_wrist",
    "dy_wrist",
    "dz_wrist",
    "dx_index",
    "dy_index",
    "dz_index",
    "speed_wrist",
]

# Combined feature columns
ALL_FEATURE_COLUMNS = FEATURE_NAMES + MOTION_FEATURE_NAMES
METADATA_COLUMNS = ["label", "raw_label", "split", "video_id", "user_id"]
FULL_HEADER = ALL_FEATURE_COLUMNS + METADATA_COLUMNS

# Mapping Jester classes to Slide Presentation Controller commands
DEFAULT_SLIDE_MAPPING = {
    "Swiping Right": "next",
    "Sliding Two Fingers Right": "next",
    "Swiping Left": "previous",
    "Sliding Two Fingers Left": "previous",
    "Stop Sign": "play_pause",
    "Thumb Up": "play_pause",
    "Thumb Down": "none",
    "No gesture": "none",
    "Doing other things": "none",
    "Zooming In With Full Hand": "none",
    "Zooming Out With Full Hand": "none",
    "Zooming In With Two Fingers": "none",
    "Zooming Out With Two Fingers": "none",
}

# Thread-local storage for MediaPipe Hands detectors
thread_local = threading.local()


def get_detector(min_confidence: float = 0.3) -> Any:
    """Return a thread-local MediaPipe Hands detector instance."""
    if not hasattr(thread_local, "detector"):
        import mediapipe as mp

        thread_local.detector = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
    return thread_local.detector


def extract_video_features(
    folder_path: str,
    min_confidence: float = 0.3,
    step: int = 3,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract 63 pose landmarks and 7 motion features from a video folder.

    Returns:
        (pose_landmarks_63, motion_features_7) or None if no hand detected.
    """
    frames = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
    if not frames:
        return None

    sampled_frames = frames[::step]
    if not sampled_frames:
        sampled_frames = frames

    hands_detector = get_detector(min_confidence=min_confidence)
    detected_entries = []

    for idx, fpath in enumerate(sampled_frames):
        img = cv2.imread(fpath)
        if img is None:
            continue
        results = hands_detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            lm_array = np.array([[p.x, p.y, p.z] for p in hand.landmark], dtype=np.float32)
            detected_entries.append((idx, lm_array))

    if not detected_entries:
        return None

    first_idx, first_lm = detected_entries[0]
    last_idx, last_lm = detected_entries[-1]
    mid_idx_pos = len(detected_entries) // 2
    _, mid_lm = detected_entries[mid_idx_pos]

    pose_features = mid_lm.reshape(-1)

    wrist_start, wrist_end = first_lm[0], last_lm[0]
    index_start, index_end = first_lm[8], last_lm[8]

    dx_wrist = float(wrist_end[0] - wrist_start[0])
    dy_wrist = float(wrist_end[1] - wrist_start[1])
    dz_wrist = float(wrist_end[2] - wrist_start[2])

    dx_index = float(index_end[0] - index_start[0])
    dy_index = float(index_end[1] - index_start[1])
    dz_index = float(index_end[2] - index_start[2])

    frame_diff = max(1, last_idx - first_idx)
    speed_wrist = float(math.sqrt(dx_wrist**2 + dy_wrist**2) / frame_diff)

    motion_features = np.array(
        [dx_wrist, dy_wrist, dz_wrist, dx_index, dy_index, dz_index, speed_wrist],
        dtype=np.float32,
    )

    return pose_features, motion_features


def process_single_item(item: Any, min_conf: float, step: int) -> dict[str, Any] | None:
    """Worker function to process one video entry."""
    raw_label = str(item.label)
    split_name = str(item.split)
    video_id = str(item.video_id)
    folder_path = str(item.folder_path)
    target_label = DEFAULT_SLIDE_MAPPING.get(raw_label, "none")

    features_pair = extract_video_features(folder_path, min_confidence=min_conf, step=step)

    if features_pair is None:
        if target_label == "none":
            zero_pose = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
            zero_motion = np.zeros(len(MOTION_FEATURE_NAMES), dtype=np.float32)
            row_dict = dict(zip(FEATURE_NAMES, zero_pose))
            row_dict.update(dict(zip(MOTION_FEATURE_NAMES, zero_motion)))
            row_dict["label"] = target_label
            row_dict["raw_label"] = raw_label
            row_dict["split"] = split_name
            row_dict["video_id"] = video_id
            row_dict["user_id"] = f"jester_{video_id}"
            return row_dict
        return None

    pose_feat, motion_feat = features_pair
    row_dict = dict(zip(FEATURE_NAMES, pose_feat))
    row_dict.update(dict(zip(MOTION_FEATURE_NAMES, motion_feat)))
    row_dict["label"] = target_label
    row_dict["raw_label"] = raw_label
    row_dict["split"] = split_name
    row_dict["video_id"] = video_id
    row_dict["user_id"] = f"jester_{video_id}"
    return row_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract landmarks and motion vectors from Jester")
    parser.add_argument("--manifest", default=r"D:\Data HCI\slide_gestures\manifest.csv", help="Path to manifest CSV")
    parser.add_argument("--output", default="data/jester_landmarks.csv", help="Output landmarks CSV file")
    parser.add_argument("--split", default="all", choices=["all", "train", "validation", "test"], help="Filter by split")
    parser.add_argument("--limit-per-class", type=int, default=None, help="Limit samples per class")
    parser.add_argument("--step", type=int, default=3, help="Sample every N-th frame (default: 3)")
    parser.add_argument("--min-conf", type=float, default=0.3, help="MediaPipe min detection confidence")
    parser.add_argument("--workers", type=int, default=6, help="Number of worker threads (default: 6)")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest không tồn tại: {manifest_path}")

    out_file = Path(args.output)
    ensure_parent(out_file)

    # Check for existing records to resume
    existing_video_ids = set()
    if out_file.exists():
        try:
            existing_df = pd.read_csv(out_file, usecols=["video_id"])
            existing_video_ids = set(existing_df["video_id"].astype(str))
            print(f"Resuming extraction: Found {len(existing_video_ids)} already extracted in {out_file}")
        except Exception:
            pass
    else:
        pd.DataFrame(columns=FULL_HEADER).to_csv(out_file, index=False)

    print("=" * 65)
    print("JESTER LANDMARK & MOTION VECTOR EXTRACTION (High-Performance)")
    print(f"Manifest:    {manifest_path}")
    print(f"Output:      {args.output}")
    print(f"Split:       {args.split}")
    print(f"Workers:     {args.workers} threads")
    print(f"Frame step:  every {args.step} frames")
    print(f"Limit/class: {args.limit_per_class or 'Full'}")
    print("=" * 65)

    df = pd.read_csv(manifest_path)
    if args.split != "all":
        df = df[df["split"] == args.split]

    if args.limit_per_class is not None:
        df = df.groupby("label", as_index=False).head(args.limit_per_class).reset_index(drop=True)

    # Filter out already extracted video IDs
    if existing_video_ids:
        initial_count = len(df)
        df = df[~df["video_id"].astype(str).isin(existing_video_ids)].reset_index(drop=True)
        print(f"Skipping {initial_count - len(df)} already processed videos.")

    total_samples = len(df)
    print(f"Videos remaining to process: {total_samples}")

    if total_samples == 0:
        print("All requested videos have already been extracted!")
        return

    items = list(df.itertuples(index=False))
    extracted_rows: list[dict[str, Any]] = []
    saved_count = len(existing_video_ids)
    skipped_count = 0
    t0 = time.time()
    processed_count = 0

    write_lock = threading.Lock()

    def task(item):
        return process_single_item(item, min_conf=args.min_conf, step=args.step)

    print(f"Starting parallel extraction with {args.workers} workers...")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(task, items, chunksize=20):
            processed_count += 1
            if result is not None:
                extracted_rows.append(result)
            else:
                skipped_count += 1

            # Flush batch to disk every 500 samples
            if len(extracted_rows) >= 500:
                with write_lock:
                    pd.DataFrame(extracted_rows).to_csv(out_file, mode="a", header=False, index=False)
                    saved_count += len(extracted_rows)
                    extracted_rows.clear()

            if processed_count % 200 == 0 or processed_count == total_samples:
                elapsed = time.time() - t0
                speed = processed_count / elapsed if elapsed > 0 else 0
                eta_s = (total_samples - processed_count) / speed if speed > 0 else 0
                current_total_saved = saved_count + len(extracted_rows)
                print(
                    f"[{processed_count:,}/{total_samples:,}] "
                    f"Saved: {current_total_saved:,} | Skipped: {skipped_count:,} | "
                    f"Speed: {speed:.1f} vid/s | ETA: {eta_s / 60:.1f} min"
                )

    # Flush remaining rows
    if extracted_rows:
        with write_lock:
            pd.DataFrame(extracted_rows).to_csv(out_file, mode="a", header=False, index=False)
            saved_count += len(extracted_rows)
            extracted_rows.clear()

    total_time = time.time() - t0
    print("=" * 65)
    print(f"EXTRACTION COMPLETED: {saved_count:,} total samples in {args.output}")
    print(f"Skipped in this run: {skipped_count:,}")
    print(f"Total time elapsed: {total_time:.2f} seconds ({total_time / 60:.1f} minutes)")
    print("=" * 65)


if __name__ == "__main__":
    main()
