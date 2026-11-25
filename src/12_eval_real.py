"""Đánh giá các model trên DỮ LIỆU TỰ THU (data/real_*.csv, xem data/REAL_DATA_README.md).

Hai phần:
  1. 6 video thuyết trình dài (real_long_landmarks.csv) chạy như LUỒNG LIÊN TỤC: model dự đoán ở mỗi bước, policy chạy
     xuyên suốt video (không reset giữa các đoạn) — giống app thật. Mỗi lệnh thực thi được gán vào đoạn nhãn chứa nó:
       đoạn intentional: trúng nếu có lệnh đúng; sai chiều nếu có lệnh ngược; độ trễ = lệnh đúng đầu tiên - đầu đoạn;
       đoạn background/near_miss: mỗi lệnh là một lần kích hoạt nhầm.
  2. 61 clip near-miss ngắn (real_short_landmarks.csv): mỗi clip một luồng giả lập (8 bước không tay -> clip -> 16 bước
     giữ tay), có kích hoạt lệnh nào không (theo nhóm tình huống và khoảng cách).

Nhịp lấy mẫu: dữ liệu thật trích ở 7.5 fps (133 ms/bước), model train ở 6 fps (167 ms/bước).
  --rate native : đưa thẳng chuỗi 7.5 fps vào model (motion mỗi bước nhỏ hơn lúc train ~20%).
  --rate 6fps   : lấy mẫu lại theo thời gian về 6 fps như SequenceBuffer của app, tính lại motion.
Cấu hình policy: tham số đã chỉnh trên Jester cho C1-C4 (reports/tuning/selected.csv, ngân sách 2%), cấu hình demo của
app (ngưỡng 0.8, nhận ngay), và quét thời gian giữ cho C1 ở ngưỡng 0.8. Mọi thời gian viết bằng ms rồi đổi ra bước.

    python src/12_eval_real.py                       # mọi model trong models/zoo, cả hai nhịp
    python src/12_eval_real.py --runs gru_L16,bigru_L16 --rates 6fps
Kết quả: reports/real/summary.csv, reports/real/events.csv, reports/real/near_miss_short.csv
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from jester_data import LANDMARK_COLUMNS
from policies import DecisionPolicy
from sequence_utils import motion_sequence, sliding_windows

COMMANDS = ["next", "previous"]
OPPOSITE = {"next": "previous", "previous": "next"}
POLICY_FLAGS = {"fixed": (False, False), "adaptive": (True, False),
                "fixed_abstention": (False, True), "adaptive_abstention": (True, True)}


# ---------- dữ liệu ----------
def resample_6fps(sec: np.ndarray, step_s: float) -> np.ndarray:
    """Chỉ số các bước được giữ khi lấy mẫu lại theo thời gian, cùng thuật toán SequenceBuffer.push của app."""
    keep, next_t = [], None
    for i, t in enumerate(sec):
        if next_t is not None and t < next_t - 1e-6:
            continue
        if next_t is None or t - next_t > step_s:
            next_t = float(t)
        next_t += step_s
        keep.append(i)
    return np.asarray(keep)


JESTER_HAND_SIZE, JESTER_CENTER = 0.169, (0.515, 0.848)  # trung vị trên Jester test (cổ tay -> gốc ngón giữa; vị trí tay)


def zoom_to_jester(landmarks: np.ndarray) -> np.ndarray:
    """Chẩn đoán: phóng to tĩnh cả video quanh vị trí tay cho kích thước tay bằng trung vị Jester (như crop camera vào
    người nói). Người 2 đứng xa nên tay nhỏ ~4.5 lần Jester; model dùng tọa độ thô nên cú vuốt nhỏ gần như không thấy."""
    present = np.any(landmarks != 0, axis=1)
    if present.sum() < 2:
        return landmarks
    lm = landmarks[present]
    size = np.median(np.hypot(lm[:, 27] - lm[:, 0], lm[:, 28] - lm[:, 1]))
    scale = float(np.clip(JESTER_HAND_SIZE / max(size, 1e-6), 1.0, 6.0))
    cx, cy = np.median(lm[:, 0]), np.median(lm[:, 1])
    out = landmarks.copy()
    out[present, 0::3] = (landmarks[present, 0::3] - cx) * scale + JESTER_CENTER[0]
    out[present, 1::3] = (landmarks[present, 1::3] - cy) * scale + JESTER_CENTER[1]
    out[present, 2::3] = landmarks[present, 2::3] * scale
    return out


def stream_features(landmarks: np.ndarray, sec: np.ndarray, rate: str, step_ms: float, zoom: bool = False):
    if zoom:
        landmarks = zoom_to_jester(landmarks)
    if rate == "6fps":
        idx = resample_6fps(sec, step_ms / 1000.0)
        landmarks, sec = landmarks[idx], sec[idx]
    return motion_sequence(landmarks), sec


def load_long(path: Path):
    d = pd.read_csv(path)
    videos = []
    for vid, g in d.groupby("video_id", sort=False):
        g = g.sort_values("step_idx")
        videos.append({"video_id": vid, "speaker": g["speaker"].iloc[0],
                       "landmarks": g[LANDMARK_COLUMNS].to_numpy(np.float32), "sec": g["sec"].to_numpy(float),
                       "trial_id": g["trial_id"].to_numpy(), "scenario": g["scenario"].to_numpy(),
                       "label": g["label"].fillna("none").to_numpy()})
    return videos


def segments_of(video) -> pd.DataFrame:
    rows, start = [], 0
    tid = video["trial_id"]
    for i in range(1, len(tid) + 1):
        if i == len(tid) or tid[i] != tid[start]:
            rows.append({"trial_id": tid[start], "scenario": video["scenario"][start], "label": video["label"][start],
                         "start_sec": video["sec"][start], "end_sec": video["sec"][i - 1]})
            start = i
    return pd.DataFrame(rows)


def load_short(path: Path):
    d = pd.read_csv(path)
    clips = []
    for vid, g in d.groupby("video_id", sort=False):
        g = g.sort_values("frame_idx")
        lm = g[LANDMARK_COLUMNS].to_numpy(np.float32)
        if (g["frame_idx"] < 0).all():
            lm = np.zeros((1, 63), np.float32)
        clips.append({"video_id": vid, "distance": g["distance"].iloc[0], "sub_category": g["sub_category"].iloc[0],
                      "near_group": g["near_group"].iloc[0], "landmarks": lm, "sec": g["sec"].to_numpy(float)})
    return clips


# ---------- model ----------
def to_command_probs(raw: np.ndarray, classes: list[str]) -> np.ndarray:
    """(T, C) xác suất model -> (T, 3) [next, previous, none]: zoom gộp vào none như app."""
    out = np.zeros((len(raw), 3), np.float32)
    for i, c in enumerate(classes):
        out[:, COMMANDS.index(c) if c in COMMANDS else 2] += raw[:, i]
    return out


def predict_stream(model, feats: np.ndarray, length: int, classes) -> np.ndarray:
    windows = sliding_windows(feats[None], length)[0]
    return to_command_probs(model.predict(windows, verbose=0, batch_size=4096), classes)


# ---------- policy ----------
def frames(ms: float, step_ms: float) -> int:
    return int(round(ms / step_ms))


def configs_for(run: str, selected: pd.DataFrame, step_ms: float, length: int, src_step_ms: float) -> list[dict]:
    """Cấu hình (đổi từ khung 6 fps sang ms rồi sang khung ở nhịp đánh giá)."""
    common = dict(reverse_block_frames=frames(2500, step_ms), repeat_block_frames=length)
    out = []
    base = run.split("_far")[0]  # model tăng cường dùng tham số đã chỉnh của model gốc cùng kiến trúc
    picks = selected[(selected.run == base) & (selected.budget == 0.02) & selected.latency_cap.isna()]
    for _, r in picks.iterrows():
        cfg = json.loads(r["config"])
        policy = cfg.pop("policy")
        for key in ("hold_frames",):
            cfg[key] = 1 + frames((cfg[key] - 1) * src_step_ms, step_ms)
        for key in ("cooldown_frames", "abstain_wait_frames", "conflict_frames"):
            if key in cfg:
                cfg[key] = max(1, frames(cfg[key] * src_step_ms, step_ms))
        cfg.update(common)
        out.append({"setting": "jester_tuned", "policy": policy, "params": cfg})
    demo = dict(base_threshold=0.8, min_threshold=0.8, hold_frames=1, cooldown_frames=frames(800, step_ms), **common)
    out.append({"setting": "app_demo (0.8, nhận ngay)", "policy": "fixed", "params": demo})
    for hold in (0, 167, 333, 500):
        p = dict(base_threshold=0.8, min_threshold=0.8, hold_frames=1 + frames(hold, step_ms),
                 cooldown_frames=frames(800, step_ms), **common)
        out.append({"setting": f"hold_sweep_{hold}ms", "policy": "fixed", "params": p})
    return out


def run_policy(probs: np.ndarray, policy_name: str, params: dict) -> list[tuple[int, str]]:
    adaptive, abstain = POLICY_FLAGS[policy_name]
    policy = DecisionPolicy(policy_name, adaptive=adaptive, abstain=abstain, **params)
    names = COMMANDS + ["none"]
    events = []
    for i, p in enumerate(probs):
        order = np.argsort(p)[::-1]
        label = names[order[0]]
        d = policy.update(label, float(p[order[0]]), float(p[order[0]] - p[order[1]]),
                          {"next": float(p[0]), "previous": float(p[1]), "none": float(p[2])})
        if d.event:
            events.append((i, d.event))
    return events


def score_long(videos_probs, policy_name, params):
    seg_rows, ev_rows = [], []
    for v, probs, sec in videos_probs:
        segs = segments_of(v)
        events = run_policy(probs, policy_name, params)
        for i, cmd in events:
            t = sec[i]
            hit = segs[(segs.start_sec <= t + 1e-6) & (segs.end_sec >= t - 1e-6)]
            seg = hit.iloc[0] if len(hit) else None
            ev_rows.append({"video_id": v["video_id"], "sec": round(float(t), 3), "event": cmd,
                            "trial_id": None if seg is None else seg.trial_id,
                            "scenario": "background" if seg is None else seg.scenario,
                            "label": "none" if seg is None else seg.label})
        for _, s in segs.iterrows():
            mine = [e for e in ev_rows if e["video_id"] == v["video_id"] and e["trial_id"] == s.trial_id]
            right = [e for e in mine if e["event"] == s.label]
            seg_rows.append({"video_id": v["video_id"], "speaker": v["speaker"], "scenario": s.scenario, "label": s.label,
                             "duration_s": s.end_sec - s.start_sec, "events": len(mine),
                             "hit": int(bool(right)), "wrong": int(any(e["event"] == OPPOSITE.get(s.label) for e in mine)),
                             "latency_s": (right[0]["sec"] - s.start_sec) if right else np.nan})
    return pd.DataFrame(seg_rows), pd.DataFrame(ev_rows)


def summarize_long(seg: pd.DataFrame) -> dict:
    cmd = seg[seg.scenario == "intentional"]
    non = seg[seg.scenario != "intentional"]
    nm = seg[seg.scenario == "near_miss"]
    minutes = non.duration_s.sum() / 60
    out = {"commands": len(cmd), "hits": int(cmd.hit.sum()), "recall": cmd.hit.mean(),
           "clean_recall": ((cmd.hit == 1) & (cmd.wrong == 0)).mean(), "wrong_dir": int(cmd.wrong.sum()),
           "false_triggers": int(non.events.sum()), "false_per_min": non.events.sum() / minutes if minutes else np.nan,
           "near_miss_long_triggered": f"{int((nm.events > 0).sum())}/{len(nm)}",
           "median_latency_s": cmd.latency_s.median()}
    for spk, g in cmd.groupby("speaker"):
        out[f"recall_{spk}"] = f"{int(g.hit.sum())}/{len(g)}"
    return out


def score_short(clips_probs, policy_name, params) -> pd.DataFrame:
    rows = []
    for c, probs in clips_probs:
        events = run_policy(probs, policy_name, params)
        rows.append({"video_id": c["video_id"], "distance": c["distance"], "sub_category": c["sub_category"],
                     "triggered": int(bool(events)), "events": ",".join(e for _, e in events)})
    return pd.DataFrame(rows)


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--long", default="data/real_long_landmarks.csv")
    parser.add_argument("--short", default="data/real_short_landmarks.csv")
    parser.add_argument("--model-dir", default="models/zoo")
    parser.add_argument("--runs", default="", help="Mặc định: mọi model trong models/zoo")
    parser.add_argument("--rates", default="native,6fps")
    parser.add_argument("--selected", default="reports/tuning/selected.csv")
    parser.add_argument("--out-dir", default="reports/real")
    parser.add_argument("--pre-frames", type=int, default=8)
    parser.add_argument("--post-frames", type=int, default=16)
    parser.add_argument("--zoom", action="store_true", help="Chẩn đoán: phóng to tay về cỡ Jester (zoom_to_jester)")
    args = parser.parse_args()

    import tensorflow as tf

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(args.selected)
    runs = [r for r in args.runs.split(",") if r] or sorted(p.stem for p in Path(args.model_dir).glob("*.keras"))
    longs, shorts = load_long(Path(args.long)), load_short(Path(args.short))
    native_step = float(np.median(np.diff(longs[0]["sec"]))) * 1000
    print(f"{len(longs)} video dài, {len(shorts)} clip near-miss ngắn, nhịp gốc {native_step:.0f} ms/bước", flush=True)

    summary, all_events, short_rows = [], [], []
    for run in runs:
        meta = json.loads((Path(args.model_dir) / f"{run}.meta.json").read_text(encoding="utf-8"))
        length, model_step, classes = int(meta["sequence_length"]), float(meta["step_ms"]), meta["classes"]
        model = tf.keras.models.load_model(Path(args.model_dir) / f"{run}.keras")
        for rate in args.rates.split(","):
            step_ms = model_step if rate == "6fps" else native_step
            vp = []
            for v in longs:
                feats, sec = stream_features(v["landmarks"], v["sec"], rate, model_step, args.zoom)
                if rate == "6fps":  # nhãn theo bước giữ lại
                    idx = resample_6fps(v["sec"], model_step / 1000.0)
                    v = {**v, "sec": v["sec"][idx], "trial_id": v["trial_id"][idx], "scenario": v["scenario"][idx],
                         "label": v["label"][idx]}
                vp.append((v, predict_stream(model, feats, length, classes), sec))
            cp = []
            for c in shorts:
                lm, sec = c["landmarks"], c["sec"] if len(c["sec"]) == len(c["landmarks"]) else np.arange(len(c["landmarks"])) * native_step / 1000
                if args.zoom:
                    lm = zoom_to_jester(lm)
                if rate == "6fps" and len(lm) > 1:
                    lm = lm[resample_6fps(sec, model_step / 1000.0)]
                stream = np.concatenate([np.zeros((args.pre_frames, 63), np.float32), lm,
                                         np.repeat(lm[-1:], args.post_frames, axis=0)])
                cp.append((c, predict_stream(model, motion_sequence(stream), length, classes)))
            src_step = 1000 / 6
            for cfg in configs_for(run, selected, step_ms, length, src_step):
                seg, ev = score_long(vp, cfg["policy"], cfg["params"])
                sh = score_short(cp, cfg["policy"], cfg["params"])
                row = {"run": run, "rate": rate, "setting": cfg["setting"], "policy": cfg["policy"],
                       **summarize_long(seg), "near_miss_short_triggered": f"{int(sh.triggered.sum())}/{len(sh)}",
                       "near_miss_short_rate": sh.triggered.mean(), "params": json.dumps(cfg["params"])}
                summary.append(row)
                if not ev.empty:
                    all_events.append(ev.assign(run=run, rate=rate, setting=cfg["setting"], policy=cfg["policy"]))
                short_rows.append(sh.assign(run=run, rate=rate, setting=cfg["setting"], policy=cfg["policy"]))
            best = [r for r in summary if r["run"] == run and r["rate"] == rate]
            print(f"{run} {rate}: " + " | ".join(f"{r['setting'][:14]}/{r['policy'][:5]} trúng {r['hits']}/{r['commands']}"
                                                 f" sai {r['wrong_dir']} nhầm {r['false_triggers']}" for r in best[:5]), flush=True)
        pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    if all_events:
        pd.concat(all_events).to_csv(out / "events.csv", index=False)
    pd.concat(short_rows).to_csv(out / "near_miss_short.csv", index=False)
    print(f"Saved {out}/summary.csv")


if __name__ == "__main__":
    main()
