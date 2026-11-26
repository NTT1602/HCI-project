"""VariantPolicy mặc định phải cho đúng chuỗi lệnh của DecisionPolicy ở cả C1-C4."""
import itertools

import numpy as np
import pytest

from policies import DecisionPolicy
from policy_variants import VariantPolicy

NAMES = ["next", "previous", "none"]


def stream(seed, n=3000):
    """Chuỗi xác suất ngẫu nhiên có tự tương quan (dễ tạo cú vuốt, lưng chừng, đổi chiều)."""
    rng = np.random.default_rng(seed)
    z = np.zeros((n, 3))
    for i in range(1, n):
        z[i] = 0.8 * z[i - 1] + rng.normal(0, 1.6, 3)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


@pytest.mark.parametrize("adaptive,abstain,seed", list(itertools.product((False, True), (False, True), range(4))))
def test_default_equals_original(adaptive, abstain, seed):
    rng = np.random.default_rng(100 + seed)
    total = 0
    for _ in range(6):
        kw = dict(base_threshold=float(rng.choice([0.6, 0.7, 0.8])), min_threshold=0.6, hold_frames=int(rng.integers(1, 5)),
                  cooldown_frames=int(rng.integers(0, 9)), margin_threshold=float(rng.choice([0.1, 0.25, 0.5])),
                  abstain_wait_frames=int(rng.integers(1, 6)), conflict_frames=int(rng.integers(1, 8)),
                  reverse_block_frames=int(rng.choice([0, 6])), repeat_block_frames=int(rng.choice([0, 6])))
        a = DecisionPolicy("x", adaptive=adaptive, abstain=abstain, **kw)
        b = VariantPolicy(adaptive=adaptive, abstain=abstain, **kw)
        got_a, got_b = [], []
        for i, p in enumerate(stream(seed)):
            o = np.argsort(p)[::-1]
            label, conf, margin = NAMES[o[0]], float(p[o[0]]), float(p[o[0]] - p[o[1]])
            ea = a.update(label, conf, margin, dict(zip(NAMES, map(float, p)))).event
            eb = b.step(None if label == "none" else label, conf, margin, float(p[0]), float(p[1]))
            got_a += [(i, ea)] if ea else []
            got_b += [(i, eb)] if eb else []
        assert got_a == got_b
        total += len(got_a)
    assert total > 50


def feed(policy, seq):
    return [policy.step(*s) for s in seq]


NEXT = ("next", 0.9, 0.82, 0.9, 0.08)
NONE = (None, 0.98, 0.96, 0.01, 0.01)


def test_noise_raises_threshold_only_when_context_is_long_enough():
    kw = dict(adaptive=True, threshold_mode="range", signal="noise", min_threshold=0.6, max_threshold=0.99, gain=4,
              hold_frames=1, cooldown_frames=0)
    history = [("previous", 0.55, 0.15, 0.4, 0.55)] * 6 + [NONE] * 8  # khua tay rồi nghỉ 8 khung
    assert feed(VariantPolicy(context_frames=5, **kw), history + [NEXT])[-1] == "next"
    assert feed(VariantPolicy(context_frames=20, **kw), history + [NEXT])[-1] is None


def test_grace_keeps_candidate_over_one_dropped_frame():
    seq = [NEXT, NEXT, NONE, NEXT]
    assert feed(VariantPolicy(hold_frames=3), seq)[-1] is None
    assert feed(VariantPolicy(hold_frames=3, grace_frames=1), seq)[-1] == "next"


def test_strict_expiry_rejects_unclear_candidate():
    unclear = ("next", 0.85, 0.75, 0.85, 0.1)
    kw = dict(abstain=True, hold_frames=1, abstain_wait_frames=2)
    assert "next" in feed(VariantPolicy(**kw), [unclear] * 4)
    assert "next" not in feed(VariantPolicy(expiry="strict", **kw), [unclear] * 8)
