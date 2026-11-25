"""Quét ngưỡng x thời gian giữ (C1) trên dữ liệu tự thu, 6 fps.

    python src/13_sweep_real_threshold.py gru_L16_far,gru_L16 reports/real_far/threshold_sweep.csv
"""
import sys, json, importlib.util
sys.path.insert(0, "src"); sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np, pandas as pd, tensorflow as tf
spec = importlib.util.spec_from_file_location("ev", "src/12_eval_real.py"); ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
from sequence_utils import motion_sequence
longs, shorts = ev.load_long(Path("data/real_long_landmarks.csv")), ev.load_short(Path("data/real_short_landmarks.csv"))
native = float(np.median(np.diff(longs[0]["sec"]))) * 1000
rows = []
for run in sys.argv[1].split(","):
    meta = json.loads(Path(f"models/zoo/{run}.meta.json").read_text(encoding="utf-8"))
    L, step, classes = int(meta["sequence_length"]), float(meta["step_ms"]), meta["classes"]
    model = tf.keras.models.load_model(f"models/zoo/{run}.keras")
    vp = []
    for v in longs:
        feats, sec = ev.stream_features(v["landmarks"], v["sec"], "6fps", step)
        idx = ev.resample_6fps(v["sec"], step / 1000)
        v = {**v, "sec": v["sec"][idx], "trial_id": v["trial_id"][idx], "scenario": v["scenario"][idx], "label": v["label"][idx]}
        vp.append((v, ev.predict_stream(model, feats, L, classes), sec))
    cp = []
    for c in shorts:
        lm = c["landmarks"]; sec = c["sec"] if len(c["sec"]) == len(lm) else np.arange(len(lm)) * native / 1000
        if len(lm) > 1: lm = lm[ev.resample_6fps(sec, step / 1000)]
        s = np.concatenate([np.zeros((8, 63), np.float32), lm, np.repeat(lm[-1:], 16, axis=0)])
        cp.append((c, ev.predict_stream(model, motion_sequence(s), L, classes)))
    for thr in (0.8, 0.9, 0.95, 0.98, 0.99):
        for hold in (0, 167, 333, 500):
            p = dict(base_threshold=thr, min_threshold=thr, hold_frames=1 + ev.frames(hold, step), cooldown_frames=ev.frames(800, step),
                     reverse_block_frames=ev.frames(2500, step), repeat_block_frames=L)
            seg, _ = ev.score_long(vp, "fixed", p); sh = ev.score_short(cp, "fixed", p); s = ev.summarize_long(seg)
            rows.append({"run": run, "thr": thr, "hold": hold, "hits": s["hits"], "n1": s["recall_nguoi_1"], "n2": s["recall_nguoi_2"],
                         "wrong": s["wrong_dir"], "false": s["false_triggers"], "nm_short": int(sh.triggered.sum())})
d = pd.DataFrame(rows); d.to_csv(sys.argv[2], index=False); print(d.to_string(index=False))
