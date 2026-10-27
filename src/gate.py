"""Reusable gesture confidence gate."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import csv


@dataclass
class GestureGate:
    threshold: float = 0.80
    hold_frames: int = 5
    cooldown_frames: int = 20
    log_path: str = "data/gesture_events.csv"

    def __post_init__(self) -> None:
        self.history: deque[str] = deque(maxlen=self.hold_frames)
        self.cooldown = 0
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)

    def update(self, label: str, confidence: float) -> str | None:
        if self.cooldown:
            self.cooldown -= 1
        if confidence < self.threshold or label in {"none", "No hand"}:
            self.history.clear()
            return None
        self.history.append(label)
        if len(self.history) < self.hold_frames or self.cooldown:
            return None
        if len(set(self.history)) == 1:
            self.cooldown = self.cooldown_frames
            self.history.clear()
            self._log(label, confidence)
            return label
        return None

    def _log(self, label: str, confidence: float) -> None:
        new_file = not Path(self.log_path).exists()
        with open(self.log_path, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if new_file:
                writer.writerow(["timestamp", "label", "confidence"])
            writer.writerow([datetime.now().isoformat(timespec="seconds"), label, f"{confidence:.4f}"])


def gesture_to_slide_action(label: str) -> str | None:
    return {"next": "NEXT SLIDE", "previous": "PREVIOUS SLIDE", "play_pause": "PLAY/PAUSE"}.get(label)
