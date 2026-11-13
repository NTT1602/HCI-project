"""Replay Jester test videos as simulated camera streams through the GRU.

Each test video becomes one trial: ``pre`` frames without a hand, the gesture,
then ``post`` frames with the hand held still. At every frame the model sees
the last ``sequence_length`` frames (exactly like 05_realtime_gru.py) and the
prediction is logged. The log feeds evaluate_policies.py, so all C1-C4
policies are compared on identical model outputs.

Example:
    python src/06_simulate_stream.py --model models/gru.keras --output data/predictions.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jester_data import load_videos, stream_windows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/jester_landmarks.csv")
    parser.add_argument("--model", default="models/gru.keras")
    parser.add_argument("--output", default="data/predictions.csv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--pre-frames", type=int, default=8, help="Frame không có tay trước cử chỉ")
    parser.add_argument("--post-frames", type=int, default=16, help="Frame giữ tay yên sau cử chỉ")
    parser.add_argument("--jitter", type=float, default=0.003, help="Nhiễu landmark (độ lệch chuẩn)")
    parser.add_argument("--max-per-label", type=int, default=None, help="Giới hạn số video mỗi raw label")
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    import tensorflow as tf

    model_path = Path(args.model)
    meta = json.loads(model_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    classes = meta["classes"]
    length = int(meta["sequence_length"])
    model = tf.keras.models.load_model(model_path)
    rng = np.random.default_rng(args.seed)

    videos = load_videos(args.data, length)
    videos = videos.subset((videos.meta["split"] == args.split).to_numpy())
    if args.max_per_label:
        keep = videos.meta.groupby("raw_label").cumcount() < args.max_per_label
        videos = videos.subset(keep.to_numpy())
    print(f"Simulating {len(videos)} trials from split '{args.split}'")
    print(videos.meta["scenario"].value_counts().to_string())

    total = args.pre_frames + length + args.post_frames
    frames = np.arange(total)
    parts = []
    for start in range(0, len(videos), args.chunk):
        chunk = videos.subset(np.arange(len(videos)) // args.chunk == start // args.chunk)
        windows = stream_windows(chunk.tracks, length, args.pre_frames, args.post_frames, args.jitter, rng)
        probs = model.predict(windows.reshape(-1, length, windows.shape[-1]), verbose=0, batch_size=2048)
        probs = probs.reshape(len(chunk), total, len(classes))
        top2 = np.sort(probs, axis=-1)[..., -2:]
        best = probs.argmax(axis=-1)
        m = chunk.meta
        class_probs = {f"p_{name}": probs[..., i].reshape(-1).round(5) for i, name in enumerate(classes)}
        parts.append(pd.DataFrame({
            "trial_id": np.repeat("jester_" + m["video_id"].to_numpy(), total),
            "trial_type": np.repeat(m["scenario"].to_numpy(), total),
            "intended_action": np.repeat(m["target"].to_numpy(), total),
            "raw_label": np.repeat(m["raw_label"].to_numpy(), total),
            "frame_idx": np.tile(frames, len(chunk)),
            "gesture_start_frame": args.pre_frames,
            "predicted_label": np.array(classes)[best].reshape(-1),
            "confidence": top2[..., 1].reshape(-1).round(5),
            "margin": (top2[..., 1] - top2[..., 0]).reshape(-1).round(5),
            **class_probs,
        }))
        print(f"  {min(start + args.chunk, len(videos))}/{len(videos)}")

    log = pd.concat(parts, ignore_index=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(output, index=False)
    print(f"Saved {len(log):,} frame predictions to: {output}")


if __name__ == "__main__":
    main()
