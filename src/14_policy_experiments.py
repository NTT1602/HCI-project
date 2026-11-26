"""Thí nghiệm tham số C2 (cửa sổ lịch sử, dải ngưỡng) và C3 (thời gian giữ/chờ, ...) trên PHIÊN DÀI liên tục.

Vì sao cần phiên dài: luồng mô phỏng của 09/10 chỉ dài 40 khung (6.7 s) và policy reset mỗi trial, nên không thử được cửa sổ
2-10 s. Ở đây ghép MỌI video của một split thành các phiên ~60 clip (ngẫu nhiên), giữa hai clip là 3-12 khung không tay;
model chạy trên luồng ghép, policy không reset trong phiên. Khung không tay = none (app cũng hủy ứng viên khi mất tay).
Zoom gộp vào none như app. Ngoài ra chạy 6 video thuyết trình tự thu + clip near-miss ngắn.

    python src/14_policy_experiments.py predict    # 1 lần: chạy model, lưu data/cache/policy_sessions/ (cần TensorFlow)
    python src/14_policy_experiments.py run        # thí nghiệm -> reports/policy_experiments/
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from policy_variants import VariantPolicy  # noqa: E402

RUNS = "bigru_L16,gru_L16,lstm_L16,tcn_L16,gru_L10"
CACHE = Path("data/cache/policy_sessions")
STEP_MS = 1000 / 6
NAMES = ["next", "previous", "none"]
CLIPS_PER_SESSION = 60


def _module(name, file):
    spec = importlib.util.spec_from_file_location(name, SRC / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- phase 1: dự đoán
def build_sessions(videos, seed):
    """-> list (landmarks (T,63), segs[(start, end, type, label)]). Mỗi đoạn = clip + khoảng không tay sau nó."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(videos))
    scen, target = videos.meta["scenario"].to_numpy(), videos.meta["target"].to_numpy()
    sessions = []
    for k in range(0, len(order), CLIPS_PER_SESSION):
        parts, segs, t = [np.zeros((8, 63), np.float32)], [], 8
        for i in order[k:k + CLIPS_PER_SESSION]:
            clip = videos.tracks[i]
            noise = rng.normal(0, 0.003, clip.shape).astype(np.float32) * np.any(clip != 0, axis=1)[:, None]
            gap = int(rng.integers(3, 13))
            parts += [clip + noise, np.zeros((gap, 63), np.float32)]
            label = target[i] if target[i] in ("next", "previous") else "none"
            kind = "intentional" if label != "none" else ("near_miss" if scen[i] != "background" else "background")
            segs.append((t, t + len(clip) + gap, kind, label))
            t += len(clip) + gap
        sessions.append((np.concatenate(parts), segs))
    return sessions


def predict_phase(args):
    import tensorflow as tf
    cmp09, ev = _module("cmp09", "09_compare_models.py"), _module("ev12", "12_eval_real.py")
    from sequence_utils import has_hand, motion_sequence
    CACHE.mkdir(parents=True, exist_ok=True)
    longs, shorts = ev.load_long(Path("data/real_long_landmarks.csv")), ev.load_short(Path("data/real_short_landmarks.csv"))
    for run in args.runs.split(","):
        meta = json.loads(Path(f"models/zoo/{run}.meta.json").read_text(encoding="utf-8"))
        length, classes = int(meta["sequence_length"]), meta["classes"]
        assert abs(float(meta["step_ms"]) - STEP_MS) < 0.1
        model = tf.keras.models.load_model(f"models/zoo/{run}.keras")
        # luôn dùng track 16 khung (2.7 s) để phiên giống hệt nhau giữa các model
        videos = cmp09.cached_videos("data/jester_landmarks_frames.csv", 16, Path("data/cache"))
        data = {}
        for split, seed in (("validation", 11), ("test", 12)):
            sub = videos.subset((videos.meta["split"] == split).to_numpy())
            data[split] = build_sessions(sub, seed)
        real = []
        for v in longs:
            idx = ev.resample_6fps(v["sec"], STEP_MS / 1000)
            vv = {**v, **{k: v[k][idx] for k in ("sec", "trial_id", "scenario", "label")}}
            tid, segs, s0 = vv["trial_id"], [], 0
            for i in range(1, len(tid) + 1):
                if i == len(tid) or tid[i] != tid[s0]:
                    segs.append((s0, i, vv["scenario"][s0], vv["label"][s0]))
                    s0 = i
            real.append((v["landmarks"][idx], segs))
        data["real"] = real
        short = []
        for c in shorts:
            lm = c["landmarks"]
            if c["sub_category"].lower().find("thu tay") >= 0 or "command_then_return" in str(c["near_group"]):
                continue  # clip chứa lệnh thật nhưng gán none (xem REAL_DATA_EVAL mục 1.3)
            if len(lm) > 1 and len(c["sec"]) == len(lm):
                lm = lm[ev.resample_6fps(c["sec"], STEP_MS / 1000)]
            lm = np.concatenate([np.zeros((8, 63), np.float32), lm, np.repeat(lm[-1:], 8, axis=0)])
            short.append((lm, [(0, len(lm), "near_miss", "none")]))
        data["real_short"] = short
        out = {}
        for name, sessions in data.items():
            probs = [ev.predict_stream(model, motion_sequence(lm), length, classes) for lm, _ in sessions]
            for p, (lm, _) in zip(probs, sessions):
                p[~has_hand(lm)] = (0.0, 0.0, 1.0)
            out[f"{name}_probs"] = np.concatenate(probs)
            out[f"{name}_len"] = np.array([len(p) for p in probs])
            rows = [(si, *s) for si, (_, segs) in enumerate(sessions) for s in segs]
            out[f"{name}_segs"] = np.array(rows, dtype=object).astype(str)
            print(f"{run} {name}: {len(sessions)} phiên, {len(out[f'{name}_probs'])} khung, {len(rows)} đoạn", flush=True)
        np.savez_compressed(CACHE / f"{run}.npz", **out)


# ---------------------------------------------------------------- phase 2: đánh giá
def load_sessions(run, name):
    z = np.load(CACHE / f"{run}.npz")
    probs, lens, segs = z[f"{name}_probs"].astype(float), z[f"{name}_len"], z[f"{name}_segs"]
    order = np.sort(probs, axis=1)
    top, conf, margin = probs.argmax(1), order[:, 2], order[:, 2] - order[:, 1]
    labels = [None if t == 2 else NAMES[t] for t in top]
    rows = list(zip(labels, conf.tolist(), margin.tolist(), probs[:, 0].tolist(), probs[:, 1].tolist()))
    ends = np.cumsum(lens)
    sessions = []
    for si, (e, n) in enumerate(zip(ends, lens)):
        mine = [(int(s), int(t), kind, lab) for i, s, t, kind, lab in segs[segs[:, 0] == str(si)]]
        sessions.append((rows[e - n:e], mine))
    return sessions


_DATA = {}


def _init(data):
    _DATA.update(data)


def clip_outcomes(cfg, sessions):
    """Mỗi đoạn một dòng: (phiên, loại, trúng, sai chiều, số lệnh, độ trễ khung, số khung)."""
    pol = VariantPolicy(**cfg)
    out, events_all = [], []
    for si, (rows, segs) in enumerate(sessions):
        pol.reset()
        step = pol.step
        ev = [(i, e) for i, e in ((i, step(*r)) for i, r in enumerate(rows)) if e]
        events_all += [(si, i, e) for i, e in ev]
        k = 0
        for s, t, kind, lab in segs:
            while k < len(ev) and ev[k][0] < s:
                k += 1
            mine = []
            while k < len(ev) and ev[k][0] < t:
                mine.append(ev[k])
                k += 1
            right = [i for i, e in mine if e == lab]
            out.append((si, kind, int(bool(right)), int(any(e != lab for _, e in mine)), len(mine),
                        right[0] - s if right else np.nan, t - s))
    return np.array([(a, {"intentional": 0, "near_miss": 1, "background": 2}[b], *c) for a, b, *c in out], float), events_all


def metrics(o):
    cmd, non = o[o[:, 1] == 0], o[o[:, 1] != 0]
    m = {"commands": len(cmd), "hits": int(cmd[:, 2].sum()), "false": int(non[:, 4].sum())}
    if len(cmd):
        m.update(recall=cmd[:, 2].mean(), clean=((cmd[:, 2] == 1) & (cmd[:, 3] == 0)).mean(), wrong=cmd[:, 3].mean(),
                 latency_ms=float(np.nanmedian(cmd[:, 5])) * STEP_MS if np.isfinite(cmd[:, 5]).any() else np.nan)
    m.update(trigger=(non[:, 4] > 0).mean(), false_per_min=non[:, 4].sum() / (non[:, 6].sum() * STEP_MS / 60000),
             near_trigger=(o[o[:, 1] == 1][:, 4] > 0).mean())
    return m


def _eval(job):
    cfg, names = job
    out = {}
    for n in names:
        o, ev = clip_outcomes(cfg, _DATA[n])
        out.update({f"{n}_{k}": v for k, v in metrics(o).items()})
        out[f"{n}_events"] = ev
    return out


def _outcomes(job):
    return clip_outcomes(job[0], _DATA[job[1]])[0]


# ---------------------------------------------------------------- cấu hình
def common(length):
    return dict(reverse_block_frames=15, repeat_block_frames=length, conflict_frames=6, uncertain_floor=0.40, cooldown_frames=5)


def ofat(length):
    """Đổi từng tham số quanh cấu hình app (conf 0.70, dải 0.70-0.95, trọng số 0.20/0.15, chờ 600 ms, clear 0.90...)."""
    rows = []
    for hold in (1, 4):
        base = dict(common(length), base_threshold=0.70, min_threshold=0.70, max_threshold=0.95, margin_weight=0.20,
                    stability_weight=0.15, stability_frames=5, margin_threshold=0.25, clear_threshold=0.90, conflict_floor=0.50,
                    abstain_wait_frames=4, hold_frames=hold)
        h = f"giữ {round((hold - 1) * STEP_MS)} ms"

        def add(group, value, policy, **kw):
            flags = {"C1": (False, False), "C2": (True, False), "C3": (False, True)}[policy]
            rows.append({"context": h, "group": group, "value": str(value), "policy": policy,
                         "cfg": {**base, "adaptive": flags[0], "abstain": flags[1], **kw}})
        add("00 gốc", "app", "C1")
        add("00 gốc", "app", "C2")
        add("00 gốc", "app", "C3")
        for ms in (833, 1000, 2000, 5000, 10000):
            f = round(ms / STEP_MS)
            add("01 C2 cửa sổ ổn định, công thức gốc (xóa lịch sử)", f"{ms} ms", "C2", stability_frames=f)
            add("02 C2 cửa sổ ổn định, KHÔNG xóa lịch sử", f"{ms} ms", "C2", threshold_mode="keep", context_frames=f)
        for lo, hi in ((0.7, 0.95), (0.75, 0.85), (0.7, 0.8), (0.6, 0.95), (0.5, 0.99), (0.3, 0.999)):
            add("03 C2 dải [min,max], công thức gốc", f"{lo}-{hi}", "C2", min_threshold=lo, max_threshold=hi)
            for b in (0.5, 0.6):
                add(f"03b C2 dải [min,max], base {b}", f"{lo}-{hi}", "C2", base_threshold=b, min_threshold=lo, max_threshold=hi)
        for mw in (0, 0.1, 0.2, 0.4, 0.8):
            add("04 C2 trọng số margin", mw, "C2", margin_weight=mw)
        for sw in (0, 0.15, 0.3, 0.6):
            add("05 C2 trọng số ổn định (gốc)", sw, "C2", stability_weight=sw)
            add("05b C2 trọng số ổn định (không xóa, 2 s)", sw, "C2", stability_weight=sw, threshold_mode="keep", context_frames=12)
        for lo, hi in ((0.75, 0.85), (0.7, 0.9), (0.7, 0.95), (0.6, 0.95), (0.6, 0.99), (0.5, 0.99)):
            for ms in (833, 1000, 2000, 5000, 10000):
                f = round(ms / STEP_MS)
                for sig, gain in (("noise", 2.0), ("ambig_mean", 2.0), ("flip", 1.0)):
                    add(f"06 C2 mới: dải {lo}-{hi}, tín hiệu {sig}", f"{ms} ms", "C2", threshold_mode="range", signal=sig, gain=gain,
                        min_threshold=lo, max_threshold=hi, context_frames=f)
        for w in (1, 2, 3, 4, 6, 9, 12):
            add("10 C3 chờ thêm", f"{round(w * STEP_MS)} ms", "C3", abstain_wait_frames=w)
        for c in (0.75, 0.8, 0.85, 0.9, 0.95, 0.98):
            add("12 C3 clear_conf", c, "C3", clear_threshold=c)
        for u in (0.34, 0.4, 0.5, 0.6, 0.7):
            add("13 C3 uncertain_floor", u, "C3", uncertain_floor=u)
        for ms in (333, 700, 1000, 2000, 5000):
            add("14 C3 cửa sổ dò mâu thuẫn", f"{ms} ms", "C3", conflict_frames=int(np.ceil(ms / STEP_MS)) + 1)
        for cf in (0.2, 0.35, 0.5, 0.7, 1.01):
            add("15 C3 conflict_floor", cf, "C3", conflict_floor=cf)
        for mg in (0, 0.25, 0.4, 0.6, 0.8):
            add("16 C3 margin", mg, "C3", margin_threshold=mg)
        for e in ("last", "mean", "strict"):
            add("17 C3 khi hết hạn", e, "C3", expiry=e)
        for g in (0, 1, 2):
            add("18 C3 dung sai khung rớt", g, "C3", grace_frames=g)
            add("18b C1 dung sai khung rớt", g, "C1", grace_frames=g)
        for b in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
            add("19 ngưỡng gốc C1", b, "C1", base_threshold=b)
            add("19b ngưỡng gốc C3", b, "C3", base_threshold=b)
    for hd in (1, 2, 3, 4, 5, 6, 7):
        for p in ("C1", "C2", "C3"):
            flags = {"C1": (False, False), "C2": (True, False), "C3": (False, True)}[p]
            rows.append({"context": "-", "group": f"11 thời gian giữ {p}", "value": f"{round((hd - 1) * STEP_MS)} ms", "policy": p,
                         "cfg": {**rows[0]["cfg"], "adaptive": flags[0], "abstain": flags[1], "hold_frames": hd}})
    return rows


THR = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99]
HOLD = [1, 2, 3, 4, 5, 6]
WIN = [5, 6, 12, 30, 60]


def families(length, n, seed=0):
    """Ngân sách tìm kiếm như nhau: mỗi họ n cấu hình ngẫu nhiên (C1 liệt kê hết)."""
    rng = np.random.default_rng(seed)
    c = common(length)
    pick = lambda xs: xs[int(rng.integers(len(xs)))]

    def legacy(keep):
        b = pick(THR)
        return dict(adaptive=True, base_threshold=b, min_threshold=b, max_threshold=max(b, pick([0.85, 0.9, 0.95, 0.99, 0.999])),
                    margin_weight=pick([0.0, 0.1, 0.2, 0.4]), stability_weight=pick([0.1, 0.15, 0.3, 0.5]),
                    **({"threshold_mode": "keep", "context_frames": pick(WIN)} if keep else {"stability_frames": pick(WIN)}))

    def ranged():
        lo = pick(THR[:-1])
        hi = pick([x for x in THR + [0.995, 0.999] if x > lo])
        return dict(adaptive=True, threshold_mode="range", min_threshold=lo, max_threshold=hi, context_frames=pick(WIN),
                    signal=pick(["noise", "ambig_mean", "flip", "ambig_now"]), gain=pick([0.5, 1.0, 2.0, 4.0, 8.0]))

    def abst(plus):
        d = dict(abstain=True, clear_threshold=pick([0.75, 0.8, 0.85, 0.9, 0.95, 0.98]), abstain_wait_frames=pick([1, 2, 3, 4, 6, 9, 12]),
                 conflict_floor=pick([0.2, 0.35, 0.5, 0.7, 1.01]), conflict_frames=pick([3, 6, 12, 30]),
                 margin_threshold=pick([0.1, 0.25, 0.5, 0.7]), uncertain_floor=pick([0.34, 0.4, 0.5, 0.6]))
        if plus:
            d.update(expiry=pick(["last", "mean", "strict"]), grace_frames=pick([0, 1, 2]))
        return d
    fam = {"C1": [dict(c, base_threshold=t, hold_frames=h) for t in THR + [0.995, 0.999] for h in HOLD]}
    fam["C1+dung sai"] = [dict(x, grace_frames=g) for x in fam["C1"] for g in (1, 2)]
    makers = {"C2 gốc": lambda: legacy(False), "C2 không xóa lịch sử": lambda: legacy(True), "C2 dải mới": ranged,
              "C3 gốc": lambda: {"base_threshold": pick(THR), **abst(False)},
              "C3 mở rộng": lambda: {"base_threshold": pick(THR), **abst(True)},
              "C4 gốc": lambda: {**legacy(False), **abst(False)}, "C4 mở rộng": lambda: {**ranged(), **abst(True)}}
    for name, mk in makers.items():
        fam[name] = [{**c, "hold_frames": pick(HOLD), **mk()} for _ in range(n)]
    return [(f, x) for f, xs in fam.items() for x in xs]


def bootstrap(outs, ref, n=2000, seed=0):
    """CI 95% ghép cặp, lấy mẫu lại theo PHIÊN. outs: {họ: outcomes}."""
    rng = np.random.default_rng(seed)
    sess = outs[ref][:, 0].astype(int)
    k = sess.max() + 1

    def per(o):
        cmd, non = o[:, 1] == 0, o[:, 1] != 0
        clean = (o[:, 2] == 1) & (o[:, 3] == 0) & cmd
        f = lambda w: np.bincount(sess, weights=w.astype(float), minlength=k)
        return f(clean), f(cmd), f(non & (o[:, 4] > 0)), f(non)
    idx = rng.integers(0, k, (n, k))
    r = per(outs[ref])
    res = {}
    for name, o in outs.items():
        p = per(o)
        row = {}
        for tag, a, b in (("clean", 0, 1), ("trigger", 2, 3)):
            d = (p[a][idx].sum(1) / p[b][idx].sum(1)) - (r[a][idx].sum(1) / r[b][idx].sum(1))
            row[f"d_{tag}"] = p[a].sum() / p[b].sum() - r[a].sum() / r[b].sum()
            row[f"d_{tag}_lo"], row[f"d_{tag}_hi"] = np.percentile(d, [2.5, 97.5])
        res[name] = row
    return res


def run_phase(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ofat_rows, sel_rows, all_rows = [], [], []
    for run in args.runs.split(","):
        length = int(run.split("_L")[1].split("_")[0])
        data = {n: load_sessions(run, n) for n in ("validation", "test", "real", "real_short")}
        with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(data,)) as pool:
            rows = ofat(length)
            res = list(pool.map(_eval, [(r["cfg"], ["validation", "real", "real_short"]) for r in rows], chunksize=4))
            ref = {(r["context"], r["policy"]): x for r, x in zip(rows, res) if r["group"] == "00 gốc"}
            for r, x in zip(rows, res):
                b = ref.get((r["context"], r["policy"]))
                for n in ("validation", "real"):
                    ev = set(x.pop(f"{n}_events"))
                    x[f"{n}_set"] = ev
                x.pop("real_short_events")
            for r, x in zip(rows, res):
                b = ref.get((r["context"], r["policy"]))
                diff = {f"{n}_changed": len(x[f"{n}_set"] ^ b[f"{n}_set"]) if b else np.nan for n in ("validation", "real")}
                ofat_rows.append({"run": run, **{k: r[k] for k in ("context", "group", "value", "policy")},
                                  **{k: v for k, v in x.items() if not k.endswith("_set")}, **diff, "cfg": json.dumps(r["cfg"])})
            pd.DataFrame(ofat_rows).to_csv(out / "ofat.csv", index=False)
            print(f"{run}: OFAT {len(rows)} cấu hình", flush=True)

            fam = families(length, args.n)
            res = list(pool.map(_eval, [(c, ["validation", "real", "real_short"]) for _, c in fam], chunksize=8))
            df = pd.DataFrame([{"run": run, "family": f, **{k: v for k, v in x.items() if not k.endswith("_events")},
                                "cfg": json.dumps(c)} for (f, c), x in zip(fam, res)])
            all_rows.append(df)
            pd.concat(all_rows).to_csv(out / "search.csv", index=False)
            for budget in (0.01, 0.02, 0.03):
                for cap in (None, 1000):
                    picks = {}
                    for f, g in df.groupby("family", sort=False):
                        ok = g[g.validation_trigger <= budget]
                        if cap:
                            ok = ok[ok.validation_latency_ms <= cap + 1e-6]
                        if len(ok):
                            picks[f] = ok.sort_values(["validation_clean", "validation_latency_ms"], ascending=[False, True]).iloc[0]
                    outs = dict(zip(picks, pool.map(_outcomes, [(json.loads(p.cfg), "test") for p in picks.values()])))
                    ci = bootstrap(outs, "C1") if "C1" in outs else {}
                    for f, p in picks.items():
                        sel_rows.append({"run": run, "budget": budget, "latency_cap": cap, "family": f,
                                         **{f"val_{k}": p[f"validation_{k}"] for k in ("clean", "trigger", "wrong", "latency_ms")},
                                         **{f"test_{k}": v for k, v in metrics(outs[f]).items()}, **ci.get(f, {}),
                                         **{k: p[k] for k in ("real_hits", "real_false", "real_wrong", "real_short_false")}, "cfg": p.cfg})
            pd.DataFrame(sel_rows).to_csv(out / "selected.csv", index=False)
            print(f"{run}: tìm kiếm {len(fam)} cấu hình xong", flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["predict", "run"])
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--n", type=int, default=1200, help="Số cấu hình ngẫu nhiên mỗi họ policy")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--out-dir", default="reports/policy_experiments")
    args = ap.parse_args()
    (predict_phase if args.phase == "predict" else run_phase)(args)


if __name__ == "__main__":
    main()
