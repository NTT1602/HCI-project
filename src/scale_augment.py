"""Tăng cường "đứng xa camera": thu nhỏ tọa độ bàn tay của Jester quanh vị trí tay rồi dịch tới chỗ khác trong khung.

Model dùng 63 tọa độ ảnh thô (sequence_utils.py), nên người đứng xa = bàn tay nhỏ hơn + cú vuốt quét quãng ngắn hơn
= đúng một phép co tọa độ. Jester: tay ~0.17 độ rộng khung, vuốt ~0.40; người 2 (dữ liệu tự thu) tay ~0.04, vuốt ~0.08
-> co ~4 lần. Co MỌI lớp (cả none/background) để model học phân biệt lệnh với cử động khi nói ở từng cỡ tay;
nếu chỉ co lệnh, mọi cử động nhỏ sẽ thành lệnh (xem thí nghiệm --zoom của 12_eval_real.py).
Motion (7 cột) được tính lại từ tọa độ mới bởi motion_sequence, không sửa ở đây.
"""
from __future__ import annotations

import numpy as np


def shrink_tracks(
    tracks: np.ndarray,
    rng: np.random.Generator,
    max_scale: float,
    min_scale: float = 1.0,
    shift: float = 0.2,
    scales: np.ndarray | None = None,
) -> np.ndarray:
    """(N, T, 63) -> bản co 1/k quanh trung bình cổ tay, k ~ log-uniform[min_scale, max_scale] mỗi video.

    Khung không có tay (toàn 0) giữ nguyên 0. ``scales`` cho k cố định (để báo cáo theo từng mức).
    """
    n, t, d = tracks.shape
    lm = tracks.reshape(n, t, 21, 3)
    present = np.any(tracks != 0, axis=-1)  # (N, T)
    if scales is None:
        scales = np.exp(rng.uniform(np.log(min_scale), np.log(max_scale), size=n))
    k = np.asarray(scales, dtype=np.float32).reshape(n, 1, 1, 1)
    count = np.maximum(present.sum(axis=1, keepdims=True), 1)
    center = (lm[:, :, 0, :2] * present[..., None]).sum(axis=1) / count  # (N, 2) trung bình cổ tay
    target = center + rng.uniform(-shift, shift, size=(n, 2))
    target[:, 0] = np.clip(target[:, 0], 0.15, 0.85)
    target[:, 1] = np.clip(target[:, 1], 0.20, 0.95)
    out = lm.copy()
    out[..., :2] = (lm[..., :2] - center[:, None, None, :]) / k + target[:, None, None, :]
    out[..., 2] = lm[..., 2] / k[..., 0]
    out[~present] = 0.0
    return out.reshape(n, t, d).astype(np.float32)
