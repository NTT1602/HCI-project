import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scale_augment import shrink_tracks  # noqa: E402


def _tracks():
    rng = np.random.default_rng(0)
    t = rng.uniform(0.3, 0.7, size=(3, 8, 63)).astype(np.float32)
    t[:, :2] = 0.0  # khung không có tay
    return t


def test_shrink_scales_spread_and_keeps_empty_frames():
    t = _tracks()
    out = shrink_tracks(t, np.random.default_rng(1), 4, scales=np.full(3, 4.0), shift=0.0)
    assert np.all(out[:, :2] == 0)
    lm, lo = t[:, 2:].reshape(3, 6, 21, 3), out[:, 2:].reshape(3, 6, 21, 3)
    spread = lambda a: a[..., 0].max(axis=(1, 2)) - a[..., 0].min(axis=(1, 2))
    np.testing.assert_allclose(spread(lo), spread(lm) / 4, rtol=1e-5)
    np.testing.assert_allclose(lo[..., 2], lm[..., 2] / 4, rtol=1e-5)


def test_scale_one_without_shift_is_identity():
    t = _tracks()
    out = shrink_tracks(t, np.random.default_rng(1), 1, scales=np.ones(3), shift=0.0)
    np.testing.assert_allclose(out, t, atol=1e-6)
