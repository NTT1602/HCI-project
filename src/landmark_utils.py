"""Shared MediaPipe hand-landmark and dataset helpers."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

def create_hands(static_image_mode: bool = False, max_num_hands: int = 1) -> Any:
    import mediapipe as mp

    return mp.solutions.hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=max_num_hands,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )


def process_frame(frame: np.ndarray, hands: Any) -> tuple[Any, None]:
    """Process one BGR frame for the landmark preview utility."""
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return results, None


def draw_landmarks(frame: np.ndarray, results: Any) -> None:
    import mediapipe as mp

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand, mp.solutions.hands.HAND_CONNECTIONS
            )
