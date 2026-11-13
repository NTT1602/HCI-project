"""Utilities for fixed-length one-hand landmark sequences.

Per-frame feature vector (70 values):
    63 landmark coordinates (21 points x xyz) + 7 motion features
    (dx/dy/dz wrist, dx/dy/dz index tip, wrist xy speed).

Motion is the displacement from the previous frame. When the hand is missing in
either frame the motion is zero, so a hand entering the view does not create a
huge fake jump. Training, stream simulation and realtime all use this rule.
"""
from __future__ import annotations

from typing import Any

import numpy as np

LANDMARK_FEATURES = 21 * 3
SEQUENCE_FEATURES = LANDMARK_FEATURES + 7
WRIST = slice(0, 3)
INDEX_TIP = slice(24, 27)


def extract_two_hand_landmarks(results: Any) -> np.ndarray:
    """Return the first detected hand's 63 landmarks padded to 70 values."""
    features = np.zeros(SEQUENCE_FEATURES, dtype=np.float32)
    if not results.multi_hand_landmarks:
        return features
    hand = results.multi_hand_landmarks[0]
    values = [[point.x, point.y, point.z] for point in hand.landmark]
    features[:LANDMARK_FEATURES] = np.asarray(values, dtype=np.float32).reshape(-1)
    return features


def has_hand(landmarks: np.ndarray) -> np.ndarray | bool:
    """True where a landmark vector (last axis >= 63) contains a detected hand."""
    return np.any(landmarks[..., :LANDMARK_FEATURES] != 0, axis=-1)


def add_motion_features(
    features: np.ndarray,
    previous: np.ndarray | None,
) -> np.ndarray:
    """Append wrist/index displacement and wrist speed to landmark features."""
    result = np.zeros(SEQUENCE_FEATURES, dtype=np.float32)
    result[:LANDMARK_FEATURES] = features[:LANDMARK_FEATURES]
    if previous is None or not has_hand(features) or not has_hand(previous):
        return result
    wrist_delta = features[WRIST] - previous[WRIST]
    index_delta = features[INDEX_TIP] - previous[INDEX_TIP]
    result[63:66] = wrist_delta
    result[66:69] = index_delta
    result[69] = float(np.linalg.norm(wrist_delta[:2]))
    return result


def motion_sequence(landmarks: np.ndarray) -> np.ndarray:
    """Vectorised ``add_motion_features`` over the time axis.

    ``landmarks`` has shape (..., T, 63); the result has shape (..., T, 70).
    """
    landmarks = np.asarray(landmarks, dtype=np.float32)
    out = np.zeros(landmarks.shape[:-1] + (SEQUENCE_FEATURES,), dtype=np.float32)
    out[..., :LANDMARK_FEATURES] = landmarks
    if landmarks.shape[-2] < 2:
        return out
    present = has_hand(landmarks)
    valid = (present[..., 1:] & present[..., :-1])[..., None]
    wrist_delta = (landmarks[..., 1:, WRIST] - landmarks[..., :-1, WRIST]) * valid
    index_delta = (landmarks[..., 1:, INDEX_TIP] - landmarks[..., :-1, INDEX_TIP]) * valid
    out[..., 1:, 63:66] = wrist_delta
    out[..., 1:, 66:69] = index_delta
    out[..., 1:, 69] = np.linalg.norm(wrist_delta[..., :2], axis=-1)
    return out


def sliding_windows(features: np.ndarray, length: int) -> np.ndarray:
    """All windows ending at each frame, left-padded with no-hand frames.

    ``features`` has shape (N, L, F); the result has shape (N, L, length, F),
    where ``result[:, t]`` is what a realtime model sees at frame ``t``.
    """
    n, total, dims = features.shape
    padded = np.concatenate(
        [np.zeros((n, length - 1, dims), dtype=features.dtype), features], axis=1
    )
    view = np.lib.stride_tricks.sliding_window_view(padded, length, axis=1)
    # sliding_window_view puts the window axis last: (N, L, F, length).
    return np.ascontiguousarray(np.moveaxis(view, -1, 2))


def pad_or_sample(sequence: list[np.ndarray], length: int) -> np.ndarray:
    """Resample a frame sequence to exactly ``length`` frames."""
    if not sequence:
        return np.zeros((length, SEQUENCE_FEATURES), dtype=np.float32)
    indices = np.linspace(0, len(sequence) - 1, length).round().astype(int)
    return np.stack([sequence[index] for index in indices])
