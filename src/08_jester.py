"""Step 8: Extract per-frame 21 hand landmarks + motion vectors from Jester dataset.

Frame-level Implementation (for GRU sequence models):
Outputs one row per sampled frame between the first and last detected hand
frame (preserving intermediate missing-hand frames as zeros so ``sequence_utils``
suppresses re-entry jumps and keeps real gesture timing).

Optimized with multi-threading and auto-resume capability.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

FEATURE_NAMES = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


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
METADATA_COLUMNS = ["label", "raw_label", "split", "video_id", "user_id", "frame_idx"]
FULL_HEADER = ALL_FEATURE_COLUMNS + METADATA_COLUMNS

# Mapping Jester classes aligned with src/jester_data.py (5-class GRU + near_miss -> none)
DEFAULT_SLIDE_MAPPING = {
    "Swiping Right": "next",
    "Sliding Two Fingers Right": "next",
    "Swiping Left": "previous",
    "Sliding Two Fingers Left": "previous",
    "Stop Sign": "none",
    "Thumb Up": "none",
    "Thumb Down": "none",
    "No gesture": "none",
    "Doing other things": "none",
    "Zooming In With Full Hand": "zoom_in",
    "Zooming Out With Full Hand": "zoom_out",
    "Zooming In With Two Fingers": "zoom_in",
    "Zooming Out With Two Fingers": "zoom_out",
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
    step: int = 2,
    min_detected: int = 1,
) -> list[tuple[int, np.ndarray, np.ndarray]] | None:
    """Extract frame-level pose landmarks and motion features from a video folder.

    Keeps all sampled frames from the first detected hand to the last detected
    hand, preserving any intermediate dropped-hand frames as all-zero rows so
    motion deltas are zeroed on re-entry (matching ``sequence_utils.py``).
    """
    frames = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
    if not frames:
        return None

    sampled_indices = list(range(0, len(frames), max(1, step)))
    hands_detector = get_detector(min_confidence=min_confidence)

    sampled_landmarks: list[tuple[int, np.ndarray | None]] = []
    detected_positions: list[int] = []

    for pos, orig_idx in enumerate(sampled_indices):
        img = cv2.imread(frames[orig_idx])
        if img is None:
            sampled_landmarks.append((orig_idx, None))
            continue
        results = hands_detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            lm_array = np.array([[p.x, p.y, p.z] for p in hand.landmark], dtype=np.float32)
            sampled_landmarks.append((orig_idx, lm_array))
            detected_positions.append(pos)
        else:
            sampled_landmarks.append((orig_idx, None))

    if len(detected_positions) < min_detected:
        return None

    start_pos = detected_positions[0]
    end_pos = detected_positions[-1]

    rows: list[tuple[int, np.ndarray, np.ndarray]] = []
    previous: np.ndarray | None = None
    for frame_idx, landmarks in sampled_landmarks[start_pos : end_pos + 1]:
        if landmarks is None:
            pose = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
            motion = np.zeros(len(MOTION_FEATURE_NAMES), dtype=np.float32)
            rows.append((frame_idx, pose, motion))
            previous = None
            continue

        if previous is None:
            motion = np.zeros(len(MOTION_FEATURE_NAMES), dtype=np.float32)
        else:
            wrist_delta = landmarks[0] - previous[0]
            index_delta = landmarks[8] - previous[8]
            motion = np.array(
                [
                    *wrist_delta,
                    *index_delta,
                    math.sqrt(float(wrist_delta[0] ** 2 + wrist_delta[1] ** 2)),
                ],
                dtype=np.float32,
            )
        rows.append((frame_idx, landmarks.reshape(-1), motion))
        previous = landmarks
    return rows


def process_single_item(item: Any, min_conf: float, step: int) -> list[dict[str, Any]] | None:
    """Worker function to process one video entry."""
    raw_label = str(item.label)
    split_name = str(item.split)
    video_id = str(item.video_id)
    folder_path = str(item.folder_path)
    target_label = DEFAULT_SLIDE_MAPPING.get(raw_label, "none")

    # Command gestures require at least 2 detected frames to form real motion
    min_detected = 1 if target_label == "none" else 2
    features = extract_video_features(
        folder_path,
        min_confidence=min_conf,
        step=step,
        min_detected=min_detected,
    )

    if features is None:
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
            row_dict["frame_idx"] = -1
            return [row_dict]
        return None

    rows = []
    for frame_idx, pose_feat, motion_feat in features:
        row_dict = dict(zip(FEATURE_NAMES, pose_feat))
        row_dict.update(dict(zip(MOTION_FEATURE_NAMES, motion_feat)))
        row_dict["label"] = target_label
        row_dict["raw_label"] = raw_label
        row_dict["split"] = split_name
        row_dict["video_id"] = video_id
        row_dict["user_id"] = f"jester_{video_id}"
        row_dict["frame_idx"] = frame_idx
        rows.append(row_dict)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frame-level landmarks and motion vectors from Jester")
    parser.add_argument("--manifest", default=r"D:\Data HCI\slide_gestures\manifest.csv", help="Path to manifest CSV")
    parser.add_argument("--output", default="data/jester_landmarks_frames.csv", help="Output frame-level landmarks CSV file")
    parser.add_argument("--split", default="all", choices=["all", "train", "validation", "test"], help="Filter by split")
    parser.add_argument("--limit-per-class", type=int, default=None, help="Limit samples per class")
    parser.add_argument("--step", type=int, default=2, help="Sample every N-th frame (default: 2 -> ~6 fps)")
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
    saved_count = 0
    if out_file.exists() and out_file.stat().st_size > 0:
        existing_header = pd.read_csv(out_file, nrows=0)
        if "frame_idx" not in existing_header.columns:
            raise ValueError(
                f"{out_file} là CSV format cũ (1 dòng/video). "
                "Hãy đổi sang --output data/jester_landmarks_frames.csv."
            )
        existing_df = pd.read_csv(out_file, usecols=["video_id"])
        saved_count = len(existing_df)
        existing_video_ids = set(existing_df["video_id"].astype(str))
        print(f"Resuming extraction: Found {len(existing_video_ids):,} videos ({saved_count:,} rows) in {out_file}")
    else:
        pd.DataFrame(columns=FULL_HEADER).to_csv(out_file, index=False)

    print("=" * 65)
    print("JESTER FRAME-LEVEL LANDMARK EXTRACTION")
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
        if args.split != "all":
            df = df.groupby("label", as_index=False).head(args.limit_per_class).reset_index(drop=True)
        else:
            limit = max(0, args.limit_per_class)
            selected_parts = []
            for _, label_df in df.groupby("label", sort=False):
                if limit == 0:
                    continue
                counts = label_df["split"].value_counts().to_dict()
                order = [s for s in ("train", "validation", "test") if s in counts]
                order.extend(s for s in counts if s not in order)
                quotas = {s: 0 for s in order}
                rem = limit
                if "train" in counts and rem > 0:
                    quotas["train"] = 1
                    rem -= 1
                eval_target = max(1, round(limit * 0.15))
                for split_name in ("validation", "test"):
                    if split_name in counts and rem > 0:
                        take = min(rem, eval_target, counts[split_name])
                        quotas[split_name] = take
                        rem -= take
                for split_name in order:
                    if rem <= 0:
                        break
                    add = min(rem, counts[split_name] - quotas[split_name])
                    quotas[split_name] += add
                    rem -= add
                for split_name in order:
                    if quotas[split_name] > 0:
                        selected_parts.append(label_df[label_df["split"] == split_name].head(quotas[split_name]))
            df = pd.concat(selected_parts, ignore_index=True) if selected_parts else df.iloc[0:0].copy()

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
                extracted_rows.extend(result)
            else:
                skipped_count += 1

            # Flush batch to disk every 500 rows
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
                    f"Saved rows: {current_total_saved:,} | Skipped vids: {skipped_count:,} | "
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
    print(f"EXTRACTION COMPLETED: {saved_count:,} total rows in {args.output}")
    print(f"Skipped videos in this run: {skipped_count:,}")
    print(f"Total time elapsed: {total_time:.2f} seconds ({total_time / 60:.1f} minutes)")
    print("=" * 65)


if __name__ == "__main__":
    main()
