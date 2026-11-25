"""Decision policies used to evaluate Midas-touch prevention.

All policies consume the same frame-level model predictions so comparisons are
fair: only the post-model decision policy changes.

C1 fixed | C2 adaptive (ngưỡng theo độ mơ hồ) | C3 fixed_abstention | C4 adaptive_abstention.

Abstention (C3/C4) — cùng thiết kế với app HCI-Slideshow-Interaction (src/decision/gate.py):
khi cử chỉ đã giữ đủ `hold_frames` mà còn nghi ngờ thì chưa thực thi, quan sát thêm tối đa `abstain_wait_frames`.
  nghi ngờ = low_conf (dưới ngưỡng nhưng >= uncertain_floor) | unclear (< clear_threshold) | low_margin (< margin_threshold)
             | conflict (chiều vuốt ngược >= conflict_floor trong `conflict_frames` khung gần nhất)
  hết nghi ngờ -> thực thi; hết hạn: đủ ngưỡng, đủ margin và chiều ngược đã hết -> thực thi; ngược lại từ chối
  và bỏ qua nhãn đó tới khi nhãn đổi.
(Bản abstention cũ chỉ đổi tên reject thành "abstain": với confidence >= 0.8 thì margin >= 0.6 nên ngưỡng margin
không bao giờ chặn — kết quả fixed và fixed_abstention trùng nhau ở mọi chỉ số.)

Heuristic của hệ thống thật, mặc định TẮT (0) để giữ nguyên kết quả cũ, bật khi muốn mô phỏng đúng app:
  reverse_block_frames: sau khi thực thi next/previous, bỏ qua chiều ngược trong ngần này khung (nhịp đưa tay về).
  repeat_block_frames:  sau khi thực thi, bỏ qua CÙNG lệnh trong ngần này khung. Mô phỏng việc app reset bộ nhớ GRU
                        sau mỗi lệnh (log dự đoán offline không chạy lại model nên phần đuôi cú vuốt vẫn còn trong cửa sổ).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

NO_LABELS = {"", "none", "No hand"}
OPPOSITE = {"next": "previous", "previous": "next"}


@dataclass
class PolicyDecision:
    event: str | None
    status: str  # accept | reject | waiting | cooldown | abstain | abstain_reject
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
    # abstention (C3/C4)
    uncertain_floor: float = 0.40
    clear_threshold: float = 0.90
    conflict_floor: float = 0.50
    conflict_frames: int = 4
    abstain_wait_frames: int = 4
    # heuristic của hệ thống thật (0 = tắt)
    reverse_block_frames: int = 0
    repeat_block_frames: int = 0

    def __post_init__(self) -> None:
        self.history: deque[str] = deque(maxlen=max(1, self.hold_frames))
        self.reset()

    def reset(self) -> None:
        self.history.clear()
        self.cooldown = 0
        self.frame = 0
        self.run_label: str | None = None
        self.run_len = 0
        self.doubt_start: int | None = None
        self.released: str | None = None
        self.last_event: str | None = None
        self.last_event_frame = -10**9
        self.evidence: deque[dict] = deque(maxlen=max(1, self.conflict_frames))

    def update(
        self,
        label: str,
        confidence: float,
        margin: float = 1.0,
        probs: dict[str, float] | None = None,
    ) -> PolicyDecision:
        """probs: xác suất từng lớp ở khung này (cần cho dò mâu thuẫn hướng của abstention)."""
        self.frame += 1
        stability_before = self._stability(label)
        threshold = self._threshold(margin, stability_before)
        if self.cooldown:
            self.cooldown -= 1
        self.evidence.append(probs or {})

        cmd = None if label in NO_LABELS else label
        if cmd != self.released:
            self.released = None
        elif cmd is not None:
            cmd = None
        since = self.frame - self.last_event_frame
        if cmd in OPPOSITE and self.last_event == OPPOSITE[cmd] and since < self.reverse_block_frames:
            cmd = None
        if cmd is not None and cmd == self.last_event and since < self.repeat_block_frames:
            cmd = None

        if not self.abstain:
            if cmd is None or confidence < threshold:
                self._clear()
                return PolicyDecision(None, "reject", threshold, 0.0)
            self.history.append(cmd)
            self._extend_run(cmd)
            return self._finish(cmd, threshold)

        if cmd is None or confidence < self.uncertain_floor:
            self._clear()
            return PolicyDecision(None, "reject", threshold, 0.0)
        if confidence < threshold:
            self.history.clear()  # khung dưới ngưỡng không cộng vào độ ổn định
        else:
            self.history.append(cmd)
        self._extend_run(cmd)
        doubts = self._doubts(cmd, confidence, margin, threshold)
        if doubts:
            if self.doubt_start is None:
                self.doubt_start = self.frame
            if self.run_len < self.hold_frames or self.frame - self.doubt_start < self.abstain_wait_frames:
                return PolicyDecision(None, "abstain", threshold, self._stability(cmd))
            if not self._resolved(cmd, confidence, margin, threshold, probs or {}):
                self._clear()
                self.released = cmd
                return PolicyDecision(None, "abstain_reject", threshold, 0.0)
        return self._finish(cmd, threshold)

    # ---------- nội bộ ----------
    def _finish(self, cmd: str, threshold: float) -> PolicyDecision:
        stability = self._stability(cmd)
        if self.run_len < self.hold_frames:
            return PolicyDecision(None, "waiting", threshold, stability)
        if self.cooldown:
            return PolicyDecision(None, "cooldown", threshold, stability)
        self.cooldown = self.cooldown_frames
        self._clear()
        self.evidence.clear()
        self.last_event, self.last_event_frame = cmd, self.frame
        return PolicyDecision(cmd, "accept", threshold, stability)

    def _clear(self) -> None:
        self.history.clear()
        self.run_label, self.run_len, self.doubt_start = None, 0, None

    def _extend_run(self, cmd: str) -> None:
        if self.run_label != cmd:
            self.run_label, self.run_len, self.doubt_start = cmd, 0, None
        self.run_len += 1

    def _doubts(self, cmd: str, confidence: float, margin: float, threshold: float) -> list[str]:
        doubts = []
        if confidence < threshold:
            doubts.append("low_conf")
        elif confidence < self.clear_threshold:
            doubts.append("unclear")
        if margin < self.margin_threshold:
            doubts.append("low_margin")
        if cmd in OPPOSITE and self._opposite_max(cmd) >= self.conflict_floor:
            doubts.append("conflict")
        return doubts

    def _opposite_max(self, cmd: str) -> float:
        opposite = OPPOSITE[cmd]
        return max((e.get(opposite, 0.0) for e in self.evidence), default=0.0)

    def _resolved(self, cmd: str, confidence: float, margin: float, threshold: float, probs: dict) -> bool:
        if confidence < threshold or margin < self.margin_threshold:
            return False
        return cmd not in OPPOSITE or probs.get(OPPOSITE[cmd], 0.0) < self.conflict_floor

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
