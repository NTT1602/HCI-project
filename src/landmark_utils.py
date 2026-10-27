"""Shared MediaPipe hand-landmark and dataset helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

FEATURE_NAMES = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
LABELS = {"0": "none", "1": "next", "2": "previous", "3": "play_pause"}


def create_hands(static_image_mode: bool = False) -> Any:
    import mediapipe as mp

    return mp.solutions.hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )


def extract_landmarks(results: Any) -> np.ndarray | None:
    """Return normalized 63-value vector, or None when no hand is detected."""
    if not results.multi_hand_landmarks:
        return None
    hand = results.multi_hand_landmarks[0]
    values = [[point.x, point.y, point.z] for point in hand.landmark]
    return np.asarray(values, dtype=np.float32).reshape(-1)


def process_frame(frame: np.ndarray, hands: Any) -> tuple[Any, np.ndarray | None]:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    return results, extract_landmarks(results)


def draw_landmarks(frame: np.ndarray, results: Any) -> None:
    import mediapipe as mp

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand, mp.solutions.hands.HAND_CONNECTIONS
            )


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_sample(path: str | Path, features: np.ndarray, label: str, user_id: str = "local") -> None:
    target = Path(path)
    ensure_parent(target)
    row = pd.DataFrame([features], columns=FEATURE_NAMES)
    row["label"] = label
    row["user_id"] = user_id
    row.to_csv(target, mode="a", header=not target.exists(), index=False)


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    data = pd.read_csv(path)
    missing = set(FEATURE_NAMES + ["label"]) - set(data.columns)
    if missing:
        raise ValueError(f"Dataset thiếu cột: {sorted(missing)}")
    return data, data[FEATURE_NAMES].to_numpy(dtype=np.float32), data["label"].to_numpy()
