"""Utilities for fixed-length two-hand landmark sequences."""
from __future__ import annotations

from typing import Any

import numpy as np

SEQUENCE_FEATURES = 21 * 3 * 2


def extract_two_hand_landmarks(results: Any) -> np.ndarray:
    """Return left/right hand landmarks, zero-filled when a hand is missing."""
    features = np.zeros(SEQUENCE_FEATURES, dtype=np.float32)
    if not results.multi_hand_landmarks:
        return features

    handedness = getattr(results, "multi_handedness", [])
    for index, hand in enumerate(results.multi_hand_landmarks[:2]):
        side = "right"
        if index < len(handedness) and handedness[index].classification:
            side = handedness[index].classification[0].label.lower()
        slot = 1 if side == "right" else 0
        values = [[point.x, point.y, point.z] for point in hand.landmark]
        start = slot * 63
        features[start : start + 63] = np.asarray(values, dtype=np.float32).reshape(-1)
    return features


def pad_or_sample(sequence: list[np.ndarray], length: int) -> np.ndarray:
    """Resample a frame sequence to exactly ``length`` frames."""
    if not sequence:
        return np.zeros((length, SEQUENCE_FEATURES), dtype=np.float32)
    indices = np.linspace(0, len(sequence) - 1, length).round().astype(int)
    return np.stack([sequence[index] for index in indices])
