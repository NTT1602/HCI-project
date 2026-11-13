"""Train the GRU gesture classifier on Jester landmark CSV data.

Classes: next, previous, zoom_in, zoom_out + none (background activity).
The background class is required for the Midas-touch study: without it the
model is forced to map every hand movement to some command.

Example:
    python src/04_train_gru.py --data data/jester_landmarks.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from jester_data import CLASSES, NONE_LABEL, VideoSet, aligned_inputs, load_videos, stream_windows


def build_training_set(
    videos: VideoSet,
    length: int,
    augment: int,
    hold_negatives: float,
    jitter: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Aligned windows + time-shifted / jittered windows + still-hand negatives."""
    inputs = [aligned_inputs(videos.tracks)]
    labels = [videos.meta["target"].to_numpy()]
    pre = post = length
    aligned_end = pre + length - 1
    max_shift = max(1, length // 4)
    for _ in range(augment):
        windows = stream_windows(videos.tracks, length, pre, post, jitter, rng)
        ends = aligned_end + rng.integers(-max_shift, max_shift + 1, size=len(videos))
        inputs.append(windows[np.arange(len(videos)), ends])
        labels.append(videos.meta["target"].to_numpy())
    if hold_negatives > 0:
        commands = np.flatnonzero(videos.meta["target"].to_numpy() != NONE_LABEL)
        chosen = rng.choice(commands, size=int(len(commands) * hold_negatives), replace=False)
        windows = stream_windows(videos.tracks[chosen], length, pre, post, jitter, rng)
        # Last frame of the stream: the window only contains the hand held still.
        inputs.append(windows[:, -1])
        labels.append(np.full(len(chosen), NONE_LABEL, dtype=object))
    return np.concatenate(inputs).astype(np.float32), np.concatenate(labels)


def build_model(tf, train_x: np.ndarray, n_classes: int):
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(train_x.reshape(-1, train_x.shape[-1]))
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=train_x.shape[1:]),
        normalizer,
        tf.keras.layers.GRU(96, return_sequences=True),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.GRU(48),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/jester_landmarks.csv")
    parser.add_argument("--model", default="models/gru.keras")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--augment", type=int, default=1, help="Số bản dịch thời gian/jitter mỗi video")
    parser.add_argument("--hold-negatives", type=float, default=0.3,
                        help="Tỉ lệ video lệnh sinh thêm mẫu 'tay giữ yên' gán nhãn none")
    parser.add_argument("--jitter", type=float, default=0.003)
    parser.add_argument("--train-near-miss", action="store_true",
                        help="Đưa Thumb Up/Down, Stop Sign vào train như none (mặc định giữ lại để test)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import tensorflow as tf

    tf.keras.utils.set_random_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    length = args.sequence_length

    print(f"Loading {args.data} ...")
    videos = load_videos(args.data, length)
    allowed = {"intentional", "background"} | ({"near_miss"} if args.train_near_miss else set())
    in_training = videos.meta["scenario"].isin(allowed).to_numpy()
    split = videos.meta["split"].to_numpy()
    train = videos.subset(in_training & (split == "train"))
    val = videos.subset(in_training & (split == "validation"))
    test = videos.subset(in_training & (split == "test"))
    near_miss_test = videos.subset((videos.meta["scenario"] == "near_miss").to_numpy() & (split == "test"))
    print(f"Videos: train={len(train)} validation={len(val)} test={len(test)} near-miss test={len(near_miss_test)}")
    print(train.meta["target"].value_counts().to_string())

    train_x, train_labels = build_training_set(train, length, args.augment, args.hold_negatives, args.jitter, rng)
    class_index = {name: i for i, name in enumerate(CLASSES)}
    train_y = np.array([class_index[label] for label in train_labels])
    val_x = aligned_inputs(val.tracks)
    val_y = val.meta["target"].map(class_index).to_numpy()
    test_x = aligned_inputs(test.tracks)
    test_y = test.meta["target"].map(class_index).to_numpy()
    print(f"Training windows: {train_x.shape}")

    weights = compute_class_weight("balanced", classes=np.arange(len(CLASSES)), y=train_y)
    model = build_model(tf, train_x, len(CLASSES))
    model.fit(
        train_x, train_y,
        validation_data=(val_x, val_y),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=dict(enumerate(weights)),
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
        verbose=2,
    )

    target = Path(args.model)
    target.parent.mkdir(parents=True, exist_ok=True)
    model.save(target)
    target.with_suffix(".labels.json").write_text(json.dumps(CLASSES), encoding="utf-8")
    meta = {
        "classes": CLASSES,
        "sequence_length": length,
        "data": args.data,
        "train_near_miss": args.train_near_miss,
        "jitter": args.jitter,
        "seed": args.seed,
    }
    target.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved model: {target}")

    write_reports(model, test_x, test_y, near_miss_test, Path(args.report_dir))


def write_reports(model, test_x, test_y, near_miss: VideoSet, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    predicted = model.predict(test_x, verbose=0).argmax(axis=1)
    labels = list(range(len(CLASSES)))
    report = classification_report(test_y, predicted, labels=labels, target_names=CLASSES, digits=3, zero_division=0)
    matrix = pd.DataFrame(
        confusion_matrix(test_y, predicted, labels=labels),
        index=[f"true_{c}" for c in CLASSES],
        columns=[f"pred_{c}" for c in CLASSES],
    )
    lines = ["Classifier on Jester test split (aligned windows)", report, matrix.to_string()]
    if len(near_miss):
        nm_pred = model.predict(aligned_inputs(near_miss.tracks), verbose=0).argmax(axis=1)
        table = pd.crosstab(near_miss.meta["raw_label"], pd.Series(np.array(CLASSES)[nm_pred], name="predicted"))
        lines += ["", "Near-miss gestures (not command classes) -> predicted class", table.to_string()]
    text = "\n\n".join(lines)
    print(text)
    (report_dir / "classifier_report.txt").write_text(text, encoding="utf-8")
    matrix.to_csv(report_dir / "confusion_matrix.csv")
    print(f"Saved reports to: {report_dir}")


if __name__ == "__main__":
    main()
