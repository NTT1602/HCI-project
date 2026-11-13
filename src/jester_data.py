"""Jester CSV loading, label/scenario mapping and stream simulation.

Two CSV formats are supported:

* video-level (current ``data/jester_landmarks.csv``): one row per video with
  the middle-frame pose and the start->end wrist/index displacement. A short
  landmark trajectory is synthesised from it (pose translated along the
  displacement), so the GRU sees real per-frame motion features.
* frame-level (``08_jester.py`` output with ``frame_idx``): several rows per
  video; the real trajectory is resampled to the sequence length.

Scenarios used for the Midas-touch study:

* intentional: Swiping Left/Right, Zooming In/Out -> next/previous/zoom_in/zoom_out
* background:  Doing other things, No gesture   -> none (seen in training)
* near_miss:   Thumb Up, Thumb Down, Stop Sign  -> none (held out of training
  by default, so they test deliberate-looking hand motion the model never saw)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sequence_utils import LANDMARK_FEATURES, has_hand, motion_sequence, sliding_windows

LANDMARK_COLUMNS = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
MOTION_COLUMNS = ["dx_wrist", "dy_wrist", "dz_wrist", "dx_index", "dy_index", "dz_index", "speed_wrist"]

COMMAND_LABELS = {
    "Swiping Right": "next",
    "Swiping Left": "previous",
    "Zooming In With Two Fingers": "zoom_in",
    "Zooming In With Full Hand": "zoom_in",
    "Zooming Out With Two Fingers": "zoom_out",
    "Zooming Out With Full Hand": "zoom_out",
}
BACKGROUND_LABELS = {"Doing other things", "No gesture"}
NEAR_MISS_LABELS = {"Thumb Up", "Thumb Down", "Stop Sign"}
NONE_LABEL = "none"
CLASSES = ["next", "none", "previous", "zoom_in", "zoom_out"]


def scenario_of(raw_label: str) -> str | None:
    if raw_label in COMMAND_LABELS:
        return "intentional"
    if raw_label in BACKGROUND_LABELS:
        return "background"
    if raw_label in NEAR_MISS_LABELS:
        return "near_miss"
    return None


def target_of(raw_label: str) -> str:
    return COMMAND_LABELS.get(raw_label, NONE_LABEL)


@dataclass
class VideoSet:
    """Landmark tracks of shape (N, T, 63) plus per-video metadata."""

    tracks: np.ndarray
    meta: pd.DataFrame  # columns: video_id, raw_label, split, scenario, target

    def subset(self, mask: np.ndarray) -> "VideoSet":
        mask = np.asarray(mask, dtype=bool)
        return VideoSet(self.tracks[mask], self.meta[mask].reset_index(drop=True))

    def __len__(self) -> int:
        return len(self.meta)


def load_videos(path: str, length: int) -> VideoSet:
    """Load a Jester landmark CSV into fixed-length landmark tracks."""
    data = pd.read_csv(path)
    data["scenario"] = data["raw_label"].map(scenario_of)
    data = data[data["scenario"].notna()].copy()
    if data.empty:
        raise ValueError(f"{path} không chứa nhãn Jester nào được dùng")
    data["target"] = data["raw_label"].map(target_of)
    data["video_id"] = data["video_id"].astype(str)
    multi_frame = "frame_idx" in data.columns and data.groupby("video_id").size().max() > 1
    if multi_frame:
        return _frame_level(data, length)
    return _video_level(data, length)


def _video_level(data: pd.DataFrame, length: int) -> VideoSet:
    pose = data[LANDMARK_COLUMNS].to_numpy(np.float32).reshape(-1, 21, 3)
    motion = data[MOTION_COLUMNS].to_numpy(np.float32)
    tracks = synthesize_tracks(pose, motion, length)
    meta = data[["video_id", "raw_label", "split", "scenario", "target"]].reset_index(drop=True)
    return VideoSet(tracks, meta)


def synthesize_tracks(pose: np.ndarray, motion: np.ndarray, length: int) -> np.ndarray:
    """Build (N, length, 63) tracks centred on ``pose`` along the displacement.

    Every landmark is translated by the wrist displacement; the index tip also
    gets its own extra displacement (captures pinch/spread in zoom gestures).
    Videos without a detected hand stay all-zero (= no hand in every frame).
    """
    steps = (np.linspace(0.0, 1.0, length, dtype=np.float32) - 0.5)[None, :, None, None]
    wrist = motion[:, None, None, 0:3]
    index_extra = motion[:, 3:6] - motion[:, 0:3]
    tracks = pose[:, None] + steps * wrist  # (N, T, 21, 3)
    tracks[:, :, 8, :] += steps[:, :, 0, :] * index_extra[:, None, :]
    present = np.any(pose.reshape(len(pose), -1) != 0, axis=1)
    tracks[~present] = 0.0
    return tracks.reshape(len(pose), length, LANDMARK_FEATURES).astype(np.float32)


def _frame_level(data: pd.DataFrame, length: int) -> VideoSet:
    tracks = []
    rows = []
    for video_id, video in data.groupby("video_id", sort=False):
        video = video.sort_values("frame_idx")
        frames = video[LANDMARK_COLUMNS].to_numpy(np.float32)
        if "frame_idx" in video and (video["frame_idx"] < 0).all():
            frames = np.zeros((1, LANDMARK_FEATURES), dtype=np.float32)
        indices = np.linspace(0, len(frames) - 1, length).round().astype(int)
        tracks.append(frames[indices])
        first = video.iloc[0]
        rows.append({
            "video_id": video_id,
            "raw_label": first.raw_label,
            "split": first.split,
            "scenario": first.scenario,
            "target": first.target,
        })
    return VideoSet(np.stack(tracks), pd.DataFrame(rows))


def build_streams(
    tracks: np.ndarray,
    pre_frames: int,
    post_frames: int,
    jitter: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate a camera stream: no hand -> gesture -> hand held still.

    Returns per-frame landmarks of shape (N, pre + T + post, 63). Gaussian
    jitter (MediaPipe-like noise) is added only to frames with a hand.
    """
    n, _, dims = tracks.shape
    pre = np.zeros((n, pre_frames, dims), dtype=np.float32)
    post = np.repeat(tracks[:, -1:, :], post_frames, axis=1)
    stream = np.concatenate([pre, tracks, post], axis=1)
    if jitter > 0:
        noise = rng.normal(0.0, jitter, stream.shape).astype(np.float32)
        stream = stream + noise * has_hand(stream)[..., None]
    return stream


def stream_windows(
    tracks: np.ndarray,
    length: int,
    pre_frames: int,
    post_frames: int,
    jitter: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Model inputs for every frame of every simulated stream: (N, L, length, 70)."""
    stream = build_streams(tracks, pre_frames, post_frames, jitter, rng)
    return sliding_windows(motion_sequence(stream), length)


def aligned_inputs(tracks: np.ndarray) -> np.ndarray:
    """One window that exactly covers each gesture: (N, T, 70)."""
    return motion_sequence(tracks)
