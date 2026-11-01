"""Decision policies used to evaluate Midas-touch prevention.

All policies consume the same frame-level model predictions so comparisons are
fair: only the post-model decision policy changes.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class PolicyDecision:
    event: str | None
    status: str
    threshold: float
    stability: float


@dataclass
class DecisionPolicy:
    """Frame-level policy with optional adaptive threshold and abstention."""

    name: str
    base_threshold: float = 0.80
    min_threshold: float = 0.70
    max_threshold: float = 0.95
    margin_weight: float = 0.20
    stability_weight: float = 0.15
    margin_threshold: float = 0.10
    hold_frames: int = 5
    cooldown_frames: int = 20
    adaptive: bool = False
    abstain: bool = False

    def __post_init__(self) -> None:
        self.history: deque[str] = deque(maxlen=self.hold_frames)
        self.cooldown = 0

    def reset(self) -> None:
        self.history.clear()
        self.cooldown = 0

    def update(
        self,
        label: str,
        confidence: float,
        margin: float = 1.0,
    ) -> PolicyDecision:
        stability_before = self._stability(label)
        threshold = self._threshold(margin, stability_before)
        if self.cooldown:
            self.cooldown -= 1

        invalid = label in {"", "none", "No hand"} or confidence < threshold
        if invalid:
            self.history.clear()
            status = "abstain" if self.abstain and label not in {"", "none", "No hand"} else "reject"
            return PolicyDecision(None, status, threshold, 0.0)

        self.history.append(label)
        stability = self._stability(label)
        if self.abstain and margin < self.margin_threshold:
            return PolicyDecision(None, "abstain", threshold, stability)
        if len(self.history) < self.hold_frames:
            return PolicyDecision(None, "waiting", threshold, stability)
        if self.cooldown:
            return PolicyDecision(None, "cooldown", threshold, stability)
        if len(set(self.history)) != 1:
            return PolicyDecision(None, "abstain", threshold, stability)

        self.cooldown = self.cooldown_frames
        self.history.clear()
        return PolicyDecision(label, "accept", threshold, stability)

    def _stability(self, label: str) -> float:
        if not self.history:
            return 0.0
        return sum(item == label for item in self.history) / len(self.history)

    def _threshold(self, margin: float, stability: float) -> float:
        if not self.adaptive:
            return self.base_threshold
        ambiguity = max(0.0, min(1.0, 1.0 - margin))
        # There is no temporal evidence on the first frame. Do not penalize it
        # as unstable, otherwise a good gesture can never enter the history.
        instability = max(0.0, min(1.0, 1.0 - stability)) if self.history else 0.0
        value = self.base_threshold + self.margin_weight * ambiguity + self.stability_weight * instability
        return max(self.min_threshold, min(self.max_threshold, value))


def make_policies(**kwargs: Any) -> dict[str, DecisionPolicy]:
    """Create the four experimental configurations with shared parameters."""
    return {
        "fixed": DecisionPolicy("fixed", adaptive=False, abstain=False, **kwargs),
        "adaptive": DecisionPolicy("adaptive", adaptive=True, abstain=False, **kwargs),
        "fixed_abstention": DecisionPolicy("fixed_abstention", adaptive=False, abstain=True, **kwargs),
        "adaptive_abstention": DecisionPolicy("adaptive_abstention", adaptive=True, abstain=True, **kwargs),
    }
