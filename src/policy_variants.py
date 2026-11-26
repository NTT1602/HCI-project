"""Biến thể mở rộng của C1-C4 (policies.py) để thí nghiệm tham số; mặc định cho kết quả GIỐNG HỆT DecisionPolicy.

Thêm so với DecisionPolicy:
  stability_frames  độ dài lịch sử ổn định của C2 (0 = hold_frames như bản gốc). Bản gốc XÓA lịch sử ở khung không hợp lệ.
  threshold_mode
    "legacy"  công thức gốc: base + margin_weight*(1-margin) + stability_weight*(1-ổn định), kẹp [min, max]
    "keep"    như trên nhưng độ ổn định = tỉ lệ cùng nhãn trong các khung LỆNH của context_frames gần nhất, không xóa lịch sử
    "range"   ngưỡng = lo + (hi-lo)*min(1, gain*rủi_ro); lo=min_threshold, hi=max_threshold (dải ngưỡng có tác dụng thật)
  signal (cho "range"), tính trên context_frames gần nhất:
    "ambig_now" 1-margin khung hiện tại | "ambig_mean" trung bình 1-margin | "noise" trung bình xác suất lệnh TRƯỚC chuỗi
    nhãn hiện tại (người nói đang khua tay nhiều hay ít) | "flip" 1 - tỉ lệ cùng nhãn trong các khung lệnh
  expiry (abstention hết hạn chờ): "last" khung hiện tại (bản gốc) | "strict" còn nghi ngờ là từ chối | "mean" trung bình
    confidence của chuỗi >= ngưỡng
  grace_frames      bỏ qua tối đa ngần này khung rớt liên tiếp mà không hủy ứng viên (0 = bản gốc)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

OPP = {"next": "previous", "previous": "next"}


@dataclass
class VariantPolicy:
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
    uncertain_floor: float = 0.40
    clear_threshold: float = 0.90
    conflict_floor: float = 0.50
    conflict_frames: int = 4
    abstain_wait_frames: int = 4
    reverse_block_frames: int = 0
    repeat_block_frames: int = 0
    stability_frames: int = 0
    threshold_mode: str = "legacy"
    context_frames: int = 5
    signal: str = "ambig_now"
    gain: float = 1.0
    expiry: str = "last"
    grace_frames: int = 0

    def __post_init__(self):
        self.history = deque(maxlen=max(1, self.stability_frames or self.hold_frames))
        self.evidence = deque(maxlen=max(1, self.conflict_frames))
        self.ctx = deque(maxlen=max(1, self.context_frames))
        self.reset()

    def reset(self):
        self.ctx.clear()
        self.evidence.clear()
        self.cooldown = self.frame = 0
        self.released = self.last_event = None
        self.last_frame = -10**9
        self.trail_label, self.trail = None, 0
        self._clear()

    def _clear(self):
        self.history.clear()
        self.run_label, self.run_len, self.doubt_start, self.conf_sum, self.gap = None, 0, None, 0.0, 0

    def _drop(self):
        """Khung rớt: trong hạn grace thì giữ ứng viên, quá hạn thì hủy."""
        if self.run_label is not None and self.gap < self.grace_frames:
            self.gap += 1
        else:
            self._clear()

    def _threshold(self, label, margin):
        if not self.adaptive:
            return self.base_threshold
        if self.threshold_mode == "range":
            risk = min(1.0, max(0.0, self.gain * self._risk(label, margin)))
            return self.min_threshold + (self.max_threshold - self.min_threshold) * risk
        ambiguity = max(0.0, min(1.0, 1.0 - margin))
        if self.threshold_mode == "keep":
            instability = self._flip(label)
        else:
            h = self.history
            instability = 1.0 - sum(x == label for x in h) / len(h) if h else 0.0
        value = self.base_threshold + self.margin_weight * ambiguity + self.stability_weight * instability
        return max(self.min_threshold, min(self.max_threshold, value))

    def _flip(self, label):
        cmds = [c[0] for c in self.ctx if c[0] is not None]
        return 1.0 - sum(x == label for x in cmds) / len(cmds) if cmds else 0.0

    def _risk(self, label, margin):
        if self.signal == "ambig_now":
            return 1.0 - margin
        if self.signal == "ambig_mean":
            return sum(c[2] for c in self.ctx) / len(self.ctx) if self.ctx else 0.0
        if self.signal == "noise":
            before = len(self.ctx) - self.trail
            return sum(self.ctx[i][1] for i in range(before)) / self.context_frames if before > 0 else 0.0
        return self._flip(label)

    def step(self, label, conf, margin, pn, pp):
        """Một khung. label: 'next' | 'previous' | None. pn/pp: xác suất next/previous. Trả lệnh thực thi hoặc None."""
        self.frame += 1
        if label == self.trail_label:
            self.trail += 1
        else:
            self.trail_label, self.trail = label, 1
        # ngữ cảnh gồm cả khung hiện tại (khung hiện tại thuộc `trail` nên không tính vào "noise")
        self.ctx.append((label, pn + pp, 1.0 - margin))
        self.evidence.append((pn, pp))
        if self.cooldown:
            self.cooldown -= 1
        if label is None:
            self.released = None
            self._drop()
            return None
        cmd = label
        thr = self._threshold(label, margin)
        if cmd != self.released:
            self.released = None
        else:
            cmd = None
        since = self.frame - self.last_frame
        if cmd is not None and self.last_event is not None:
            if self.last_event == OPP[cmd]:
                if since < self.reverse_block_frames:
                    cmd = None
            elif since < self.repeat_block_frames:
                cmd = None
        if cmd is None:
            self._clear()
            return None
        if not self.abstain:
            if conf < thr:
                self._drop()
                return None
            self._extend(cmd, conf)
            self.history.append(cmd)
            return self._finish(cmd)
        if conf < self.uncertain_floor:
            self._drop()
            return None
        if conf < thr:
            self.history.clear()
        else:
            self.history.append(cmd)
        self._extend(cmd, conf)
        opp = max((e[1] if cmd == "next" else e[0]) for e in self.evidence)
        if conf < max(thr, self.clear_threshold) or margin < self.margin_threshold or opp >= self.conflict_floor:
            if self.doubt_start is None:
                self.doubt_start = self.frame
            if self.run_len < self.hold_frames or self.frame - self.doubt_start < self.abstain_wait_frames:
                return None
            if self.expiry == "strict":
                ok = False
            else:
                c = conf if self.expiry == "last" else self.conf_sum / self.run_len
                ok = c >= thr and margin >= self.margin_threshold and (pp if cmd == "next" else pn) < self.conflict_floor
            if not ok:
                self._clear()
                self.released = cmd
                return None
        return self._finish(cmd)

    def _extend(self, cmd, conf):
        if self.run_label != cmd:
            self.run_label, self.run_len, self.doubt_start, self.conf_sum = cmd, 0, None, 0.0
        self.run_len += 1
        self.conf_sum += conf
        self.gap = 0

    def _finish(self, cmd):
        if self.run_len < self.hold_frames or self.cooldown:
            return None
        self.cooldown = self.cooldown_frames
        self._clear()
        self.evidence.clear()
        self.last_event, self.last_frame = cmd, self.frame
        return cmd
