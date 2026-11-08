"""Step 8: Extract 21 hand landmarks + motion vectors from Jester dataset.

Option 2 Implementation:
Combines 63 static landmarks (at keyframe) with dynamic motion vectors (dx, dy, dz)
between gesture start and end frames to effectively distinguish directional gestures
(such as Swiping Left vs. Swiping Right).

Usage:
    # Quick test run with 50 samples per class:
    python src/08_jester.py --limit-per-class 50 --output data/jester_sample.csv

    # Full extraction:
    python src/08_jester.py --output data/jester_landmarks.csv
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time
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


def create_jester_hands(min_confidence: float = 0.3) -> Any:
    """Initialize MediaPipe Hands optimized for low-resolution Jester frames (100px)."""
    import mediapipe as mp

    return mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=min_confidence,
        min_tracking_confidence=min_confidence,
    )


def extract_video_features(
    folder_path: str,
    hands_detector: Any,
    step: int = 2,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract 63 pose landmarks and 7 motion features from a video folder.

    Returns:
        (pose_landmarks_63, motion_features_7) or None if no hand detected.
    """
    frames = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
    if not frames:
        return None

    # Sample frames according to step
    sampled_frames = frames[::step]
    if not sampled_frames:
        sampled_frames = frames

    detected_entries = []

    for idx, fpath in enumerate(sampled_frames):
        img = cv2.imread(fpath)
        if img is None:
            continue
        results = hands_detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            # Convert to numpy array of 21 x 3
            lm_array = np.array([[p.x, p.y, p.z] for p in hand.landmark], dtype=np.float32)
            detected_entries.append((idx, lm_array))

    if not detected_entries:
        return None

    # Start, middle, and end frames with hands
    first_idx, first_lm = detected_entries[0]
    last_idx, last_lm = detected_entries[-1]
    mid_idx_pos = len(detected_entries) // 2
    _, mid_lm = detected_entries[mid_idx_pos]

    # 63 static pose features at keyframe (midpoint of gesture)
    pose_features = mid_lm.reshape(-1)

    # 7 motion features:
    # Landmark 0 = Wrist
    # Landmark 8 = Index finger tip
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract landmarks and motion vectors from Jester")
    parser.add_argument("--manifest", default=r"D:\Data HCI\slide_gestures\manifest.csv", help="Path to manifest CSV")
    parser.add_argument("--output", default="data/jester_landmarks.csv", help="Output landmarks CSV file")
    parser.add_argument("--split", default="all", choices=["all", "train", "validation", "test"], help="Filter by split")
    parser.add_argument("--limit-per-class", type=int, default=None, help="Limit samples per class for quick testing")
    parser.add_argument("--step", type=int, default=2, help="Sample every N-th frame in each video (default: 2)")
    parser.add_argument("--min-conf", type=float, default=0.3, help="MediaPipe min detection confidence")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest không tồn tại: {manifest_path}")

    print("=" * 65)
    print("JESTER LANDMARK & MOTION VECTOR EXTRACTION (Option 2)")
    print(f"Manifest: {manifest_path}")
    print(f"Output:   {args.output}")
    print(f"Split:    {args.split}")
    print(f"Limit/class: {args.limit_per_class or 'Full'}")
    print("=" * 65)

    df = pd.read_csv(manifest_path)
    if args.split != "all":
        df = df[df["split"] == args.split]

    # Balance/limit per class if requested
    if args.limit_per_class is not None:
        df = df.groupby("label", as_index=False).head(args.limit_per_class).reset_index(drop=True)

    total_samples = len(df)
    print(f"Total videos to process: {total_samples}")
    ensure_parent(args.output)

    # Initialize CSV header if output doesn't exist
    out_file = Path(args.output)
    header_columns = (
        ALL_FEATURE_COLUMNS
        + ["label", "raw_label", "split", "video_id", "user_id"]
    )
    pd.DataFrame(columns=header_columns).to_csv(out_file, index=False)

    extracted_rows: list[dict[str, Any]] = []
    saved_count = 0
    skipped_count = 0
    t0 = time.time()

    hands_detector = create_jester_hands(min_confidence=args.min_conf)

    try:
        for idx, row in enumerate(df.itertuples(index=False), start=1):
            raw_label = str(row.label)
            split_name = str(row.split)
            video_id = str(row.video_id)
            folder_path = str(row.folder_path)
            target_label = DEFAULT_SLIDE_MAPPING.get(raw_label, "none")

            features_pair = extract_video_features(folder_path, hands_detector, step=args.step)

            if features_pair is None:
                # For negative / background classes (No gesture / Doing other things),
                # an absence of hands is a valid negative sample!
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
                    extracted_rows.append(row_dict)
                    saved_count += 1
                else:
                    skipped_count += 1
                continue

            pose_feat, motion_feat = features_pair
            row_dict = dict(zip(FEATURE_NAMES, pose_feat))
            row_dict.update(dict(zip(MOTION_FEATURE_NAMES, motion_feat)))
            row_dict["label"] = target_label
            row_dict["raw_label"] = raw_label
            row_dict["split"] = split_name
            row_dict["video_id"] = video_id
            row_dict["user_id"] = f"jester_{video_id}"
            extracted_rows.append(row_dict)
            saved_count += 1

            # Batch write to disk every 200 samples
            if len(extracted_rows) >= 200:
                pd.DataFrame(extracted_rows).to_csv(out_file, mode="a", header=False, index=False)
                extracted_rows.clear()

            if idx % 50 == 0 or idx == total_samples:
                elapsed = time.time() - t0
                speed = idx / elapsed if elapsed > 0 else 0
                eta_s = (total_samples - idx) / speed if speed > 0 else 0
                print(
                    f"[{idx}/{total_samples}] Processed | Saved: {saved_count} | Skipped: {skipped_count} | "
                    f"Speed: {speed:.1f} vid/s | ETA: {eta_s / 60:.1f} min"
                )

        # Flush remaining rows
        if extracted_rows:
            pd.DataFrame(extracted_rows).to_csv(out_file, mode="a", header=False, index=False)
            extracted_rows.clear()

    finally:
        hands_detector.close()

    total_time = time.time() - t0
    print("=" * 65)
    print(f"EXTRACTION COMPLETE: {saved_count} samples saved to {args.output}")
    print(f"Skipped (hands not found in positive classes): {skipped_count}")
    print(f"Total time elapsed: {total_time:.2f} seconds ({total_time / 60:.1f} minutes)")
    print("=" * 65)


if __name__ == "__main__":
    main()
