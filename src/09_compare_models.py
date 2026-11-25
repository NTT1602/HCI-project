"""So sánh nhiều kiến trúc trên CÙNG dữ liệu, cùng seed, cùng augment (src/model_zoo.py).

Với mỗi (model, độ dài cửa sổ):
  1. Train như 04_train_gru.py (aligned + dịch thời gian/jitter + mẫu "tay giữ yên" = none, class weight, early stopping).
  2. Báo cáo phân loại trên test (aligned windows) + near-miss.
  3. Đo tốc độ suy luận 1 cửa sổ (tf.function, giống app) và số tham số.
  4. Mô phỏng luồng camera (giống 06_simulate_stream.py) trên validation (để chỉnh tham số policy) và test (để báo cáo cuối),
     ghi log dự đoán từng khung cho 10_tune_policies.py.

Ví dụ:
    python src/09_compare_models.py                                   # mọi model, cửa sổ 16
    python src/09_compare_models.py --runs gru:16,gru:12,gru:10,tcn:16 --epochs 30
Kết quả: models/zoo/<model>_L<T>.keras (+ .labels.json, .meta.json — copy sang app như gru.keras),
         reports/models/classifier_summary.csv, reports/models/<run>_report.txt, data/zoo_predictions/<run>_<split>.csv
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight

from jester_data import CLASSES, VideoSet, aligned_inputs, load_videos, stream_windows
from model_zoo import MODELS, build
from scale_augment import shrink_tracks

SRC = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("train_gru", SRC / "04_train_gru.py")
train_gru = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_gru)  # dùng lại build_training_set để mọi model có cùng dữ liệu train như GRU gốc


def cached_videos(csv: str, length: int, cache_dir: Path) -> VideoSet:
    """load_videos mất ~75 s trên CSV 490 MB; lưu lại dạng npz + csv theo độ dài cửa sổ."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = cache_dir / f"{Path(csv).stem}_L{length}"
    if stem.with_suffix(".npz").exists():
        tracks = np.load(stem.with_suffix(".npz"))["tracks"]
        meta = pd.read_csv(stem.with_suffix(".meta.csv"), dtype={"video_id": str}, keep_default_na=False)
        return VideoSet(tracks, meta, frame_level=True)
    videos = load_videos(csv, length)
    np.savez_compressed(stem.with_suffix(".npz"), tracks=videos.tracks)
    videos.meta.to_csv(stem.with_suffix(".meta.csv"), index=False)
    return videos


def simulate(model, videos: VideoSet, length: int, pre: int, post: int, jitter: float, seed: int,
             chunk: int = 256) -> pd.DataFrame:
    """Bản sao logic của 06_simulate_stream.py (cùng cột) để mọi model được mô phỏng y hệt nhau."""
    rng = np.random.default_rng(seed)
    total = pre + length + post
    frames = np.arange(total)
    parts = []
    for start in range(0, len(videos), chunk):
        idx = np.arange(start, min(start + chunk, len(videos)))
        sub = videos.subset(np.isin(np.arange(len(videos)), idx))
        windows = stream_windows(sub.tracks, length, pre, post, jitter, rng)
        probs = model.predict(windows.reshape(-1, length, windows.shape[-1]), verbose=0, batch_size=4096)
        probs = probs.reshape(len(sub), total, len(CLASSES))
        top2 = np.sort(probs, axis=-1)[..., -2:]
        best = probs.argmax(axis=-1)
        m = sub.meta
        parts.append(pd.DataFrame({
            "trial_id": np.repeat("jester_" + m["video_id"].astype(str).to_numpy(), total),
            "trial_type": np.repeat(m["scenario"].to_numpy(), total),
            "intended_action": np.repeat(m["target"].to_numpy(), total),
            "raw_label": np.repeat(m["raw_label"].to_numpy(), total),
            "frame_idx": np.tile(frames, len(sub)),
            "gesture_start_frame": pre,
            "predicted_label": np.array(CLASSES)[best].reshape(-1),
            "confidence": top2[..., 1].reshape(-1).round(5),
            "margin": (top2[..., 1] - top2[..., 0]).reshape(-1).round(5),
            **{f"p_{c}": probs[..., i].reshape(-1).round(5) for i, c in enumerate(CLASSES)},
        }))
    return pd.concat(parts, ignore_index=True)


def inference_ms(tf, model, length: int, repeats: int = 300) -> float:
    """Thời gian 1 lần dự đoán 1 cửa sổ qua tf.function — đúng cách app gọi model (bất biến 6 trong CLAUDE.md của app)."""
    graph = tf.function(lambda x: model(x, training=False), reduce_retracing=True,
                        input_signature=[tf.TensorSpec((1, length, 70), tf.float32)])
    x = tf.constant(np.zeros((1, length, 70), np.float32))
    for _ in range(20):
        graph(x).numpy()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        graph(x).numpy()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def per_label_subset(videos: VideoSet, max_per_label: int | None, seed: int) -> VideoSet:
    if not max_per_label:
        return videos
    order = np.random.default_rng(seed).permutation(len(videos))
    shuffled = videos.meta.iloc[order]
    keep = order[(shuffled.groupby("raw_label").cumcount() < max_per_label).to_numpy()]
    return videos.subset(np.isin(np.arange(len(videos)), keep))


def run_one(tf, name: str, length: int, args, step_ms: float) -> dict:
    run = f"{name}_L{length}" + (f"_{args.tag}" if args.tag else "")
    print(f"\n========== {run} ==========", flush=True)
    tf.keras.utils.set_random_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    videos = cached_videos(args.data, length, Path(args.cache_dir))
    scen, split = videos.meta["scenario"].to_numpy(), videos.meta["split"].to_numpy()
    in_training = np.isin(scen, ["intentional", "background"])
    train = videos.subset(in_training & (split == "train"))
    val = videos.subset(in_training & (split == "validation"))
    test = videos.subset(in_training & (split == "test"))
    near_miss_test = videos.subset((scen == "near_miss") & (split == "test"))

    index = {c: i for i, c in enumerate(CLASSES)}
    if args.far_scale > 1:
        # Bản gốc chỉ giữ cửa sổ khớp cử chỉ; bản co (đứng xa) đi qua đủ augment (dịch thời gian, jitter, tay giữ yên).
        # Giữ tổng kích thước gần như cũ vì RAM máy nhóm hạn chế.
        orig_x, orig_labels = aligned_inputs(train.tracks), train.meta["target"].to_numpy()
        far = VideoSet(shrink_tracks(train.tracks, rng, args.far_scale), train.meta, True)
        far_x, far_labels = train_gru.build_training_set(far, length, args.augment, args.hold_negatives, args.jitter, rng)
        train_x, train_labels = np.concatenate([orig_x, far_x]), np.concatenate([orig_labels, far_labels])
        del far, far_x, orig_x
        val_far = shrink_tracks(val.tracks, np.random.default_rng(args.seed + 1), args.far_scale)
        val_x = np.concatenate([aligned_inputs(val.tracks), aligned_inputs(val_far)])
        val_y = np.tile(val.meta["target"].map(index).to_numpy(), 2)
    else:
        train_x, train_labels = train_gru.build_training_set(train, length, args.augment, args.hold_negatives, args.jitter, rng)
        val_x, val_y = aligned_inputs(val.tracks), val.meta["target"].map(index).to_numpy()
    train_y = np.array([index[x] for x in train_labels])
    test_x, test_y = aligned_inputs(test.tracks), test.meta["target"].map(index).to_numpy()

    model = build(tf, name, train_x, len(CLASSES))
    weights = compute_class_weight("balanced", classes=np.arange(len(CLASSES)), y=train_y)
    t0 = time.time()
    hist = model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=args.epochs, batch_size=args.batch_size,
                     class_weight=dict(enumerate(weights)), verbose=2,
                     callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])
    train_s = time.time() - t0

    out = Path(args.model_dir) / f"{run}.keras"
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    out.with_suffix(".labels.json").write_text(json.dumps(CLASSES), encoding="utf-8")
    out.with_suffix(".meta.json").write_text(json.dumps({
        "classes": CLASSES, "sequence_length": length, "step_ms": step_ms, "frame_level_data": True,
        "data": args.data, "architecture": name, "train_near_miss": False, "jitter": args.jitter, "seed": args.seed,
        "far_scale": args.far_scale,
    }, indent=2), encoding="utf-8")

    pred = model.predict(test_x, verbose=0, batch_size=4096).argmax(axis=1)
    nm_pred = model.predict(aligned_inputs(near_miss_test.tracks), verbose=0, batch_size=4096).argmax(axis=1)
    nm_names = np.array(CLASSES)[nm_pred]
    f1 = f1_score(test_y, pred, labels=list(range(len(CLASSES))), average=None, zero_division=0)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    text = [f"{run}: test aligned windows",
            classification_report(test_y, pred, labels=list(range(len(CLASSES))), target_names=CLASSES, digits=3, zero_division=0),
            "Near-miss -> predicted",
            pd.crosstab(near_miss_test.meta["raw_label"], pd.Series(nm_names, name="predicted")).to_string()]
    scale_acc = {}
    for k in (1, 2, 3, 4, 5):  # độ chính xác khi tay nhỏ đi k lần (test Jester co cố định)
        tk = test.tracks if k == 1 else shrink_tracks(test.tracks, np.random.default_rng(k), k, scales=np.full(len(test), k))
        acc = float((model.predict(aligned_inputs(tk), verbose=0, batch_size=4096).argmax(axis=1) == test_y).mean())
        scale_acc[f"test_acc_x{k}"] = round(acc, 4)
    text.append("Accuracy theo mức co tay (đứng xa): " + json.dumps(scale_acc))
    (report_dir / f"{run}_report.txt").write_text("\n\n".join(text), encoding="utf-8")

    row = {
        "run": run, "architecture": name, "sequence_length": length, "window_ms": round(length * step_ms),
        "params": int(model.count_params()), "epochs_run": len(hist.history["loss"]), "train_s": round(train_s),
        "infer_ms": round(inference_ms(tf, model, length), 2),
        "test_accuracy": round(float((pred == test_y).mean()), 4),
        "test_macro_f1": round(float(f1.mean()), 4),
        **{f"f1_{c}": round(float(f1[i]), 4) for i, c in enumerate(CLASSES)},
        "near_miss_to_swipe": round(float(np.isin(nm_names, ["next", "previous"]).mean()), 4),
        "near_miss_to_command": round(float((nm_names != "none").mean()), 4),
        **scale_acc,
    }
    print(row, flush=True)

    if args.skip_sim:
        return row
    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    for split_name, max_per_label in (("validation", args.val_per_label), ("test", None)):
        sub = videos.subset(split == split_name)
        sub = per_label_subset(sub, max_per_label, args.seed)
        log = simulate(model, sub, length, args.pre_frames, args.post_frames, args.jitter, args.sim_seed)
        log.to_csv(pred_dir / f"{run}_{split_name}.csv", index=False)
        print(f"  simulated {split_name}: {sub.meta.shape[0]} trials", flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/jester_landmarks_frames.csv")
    parser.add_argument("--runs", default=",".join(f"{m}:16" for m in MODELS),
                        help="Danh sách model:độ_dài, vd gru:16,gru:10,tcn:16")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--augment", type=int, default=1)
    parser.add_argument("--hold-negatives", type=float, default=0.3)
    parser.add_argument("--jitter", type=float, default=0.003)
    parser.add_argument("--step-ms", type=float, default=1000 / 6, help="Jester 12 fps lấy 1/2 khung = 6 fps")
    parser.add_argument("--pre-frames", type=int, default=8)
    parser.add_argument("--post-frames", type=int, default=16)
    parser.add_argument("--val-per-label", type=int, default=200, help="Giới hạn video/nhãn khi mô phỏng validation (chỉnh tham số)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sim-seed", type=int, default=7)
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--model-dir", default="models/zoo")
    parser.add_argument("--report-dir", default="reports/models")
    parser.add_argument("--pred-dir", default="data/zoo_predictions")
    parser.add_argument("--far-scale", type=float, default=0, help="> 1: thêm bản co tay 1..k lần (đứng xa), xem scale_augment.py")
    parser.add_argument("--tag", default="", help="Hậu tố tên run, vd far -> gru_L16_far")
    parser.add_argument("--skip-sim", action="store_true", help="Bỏ mô phỏng Jester val/test (chỉ cần khi chỉnh tham số policy)")
    args = parser.parse_args()

    import tensorflow as tf

    summary_path = Path(args.report_dir) / "classifier_summary.csv"
    rows = pd.read_csv(summary_path).to_dict("records") if summary_path.exists() else []
    for item in args.runs.split(","):
        name, length = item.split(":")
        row = run_one(tf, name.strip(), int(length), args, args.step_ms)
        rows = [r for r in rows if r["run"] != row["run"]] + [row]
        pd.DataFrame(rows).to_csv(summary_path, index=False)  # ghi sau mỗi model: chạy dở vẫn giữ kết quả
    print("\n" + pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
