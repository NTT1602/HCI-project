"""Chỉnh tham số C1-C4 cho đúng thực nghiệm, trên validation; báo cáo một lần trên test.

Cho khớp với hệ thống thật (app HCI-Slideshow-Interaction):
  * chỉ next/previous là lệnh, zoom gộp vào none và video zoom là near-miss (--commands);
  * mọi tham số thời gian viết bằng ms rồi đổi sang khung theo step_ms của model (6 fps);
  * chặn chiều ngược 2.5 s và "reset GRU sau lệnh" (repeat_block = độ dài cửa sổ) bật cho MỌI cấu hình.

Cách chọn (công bằng giữa các policy): với mỗi ngân sách kích hoạt nhầm B (tỉ lệ trial không-phải-lệnh có >= 1 lệnh,
gồm background + near-miss), chọn cấu hình có clean_success cao nhất trên validation (đúng lệnh, không kèm lệnh sai),
hòa thì latency thấp hơn. Cấu hình đã chọn chạy MỘT lần trên test. Kèm theo: tham số app đang dùng (chưa chỉnh).

Ví dụ:
    python src/10_tune_policies.py                         # mọi run có sẵn trong data/zoo_predictions
    python src/10_tune_policies.py --runs gru_L16,tcn_L16 --budgets 0.02,0.05,0.1
Kết quả: reports/tuning/<run>_val_grid.csv, reports/tuning/selected.csv, reports/tuning/app_defaults.csv
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from evaluate_policies import evaluate_prepared, prepare_trial, restrict_commands, summarize
from policies import DecisionPolicy

POLICY_FLAGS = {"fixed": (False, False), "adaptive": (True, False),
                "fixed_abstention": (False, True), "adaptive_abstention": (True, True)}
METRICS = ["clean_success_rate", "command_success_rate", "wrong_command_rate", "command_miss_rate",
           "non_command_trigger_rate", "background_trigger_rate", "near_miss_trigger_rate",
           "false_triggers_per_minute", "median_latency_ms", "abstention_rate"]

_TRIALS = None
_FPS = None


def _init(trials, fps):
    global _TRIALS, _FPS
    _TRIALS, _FPS = trials, fps


def _evaluate(config: dict) -> dict:
    adaptive, abstain = POLICY_FLAGS[config["policy"]]
    params = {k: v for k, v in config.items() if k != "policy"}
    policy = DecisionPolicy(config["policy"], adaptive=adaptive, abstain=abstain, **params)
    results = pd.DataFrame([evaluate_prepared(t, policy, _FPS)[0] for t in _TRIALS]).assign(policy=config["policy"])
    row = summarize(results).iloc[0]
    return {**config, **{m: row[m] for m in METRICS}}


def _trial_outcomes(config: dict) -> dict:
    """Kết quả từng trial (để bootstrap): đúng lệnh sạch (trial lệnh) và có kích hoạt (trial không phải lệnh)."""
    adaptive, abstain = POLICY_FLAGS[config["policy"]]
    params = {k: v for k, v in config.items() if k != "policy"}
    policy = DecisionPolicy(config["policy"], adaptive=adaptive, abstain=abstain, **params)
    res = [evaluate_prepared(t, policy, _FPS)[0] for t in _TRIALS]
    return {"clean": [r["clean_success"] for r in res if r["trial_type"] == "intentional"],
            "trigger": [int(r["events"] > 0) for r in res if r["trial_type"] != "intentional"]}


def bootstrap(outcomes: dict[str, dict], reference: str, n: int = 2000, seed: int = 0) -> list[dict]:
    """CI 95% cho từng policy và cho chênh lệch ghép cặp (policy - reference) trên cùng trial."""
    import numpy as np
    rng = np.random.default_rng(seed)
    rows = []
    ref = {k: np.asarray(v, float) for k, v in outcomes[reference].items()}
    idx = {k: rng.integers(0, len(v), size=(n, len(v))) for k, v in ref.items()}
    for policy, out in outcomes.items():
        row = {"policy": policy}
        for key in ("clean", "trigger"):
            x = np.asarray(out[key], float)
            means = x[idx[key]].mean(axis=1)
            diffs = (x - ref[key])[idx[key]].mean(axis=1)
            row[f"{key}_mean"] = x.mean()
            row[f"{key}_lo"], row[f"{key}_hi"] = np.percentile(means, [2.5, 97.5])
            row[f"{key}_diff"] = (x - ref[key]).mean()
            row[f"{key}_diff_lo"], row[f"{key}_diff_hi"] = np.percentile(diffs, [2.5, 97.5])
        rows.append(row)
    return rows


def frames(ms: float, step_ms: float) -> int:
    return int(math.ceil(ms / step_ms - 1e-9))


def load(path: Path, commands: list[str]):
    data = pd.read_csv(path, keep_default_na=False)
    data = restrict_commands(data, commands)
    return [prepare_trial(t) for _, t in data.groupby("trial_id", sort=False)]


def grid(step_ms: float, length: int, top_adaptive: list[dict] | None = None):
    """Lưới tham số, thời gian tính theo BƯỚC model (1 bước = step_ms, 167 ms ở 6 fps).
    Heuristic của app cố định cho mọi cấu hình."""
    common = dict(reverse_block_frames=frames(2500, step_ms), repeat_block_frames=length,
                  conflict_frames=frames(700, step_ms) + 1, uncertain_floor=0.40)
    # giữ 0/1/2/3 bước (0-500 ms); cooldown 5/9 bước (~0.8 s / 1.5 s)
    shared = [dict(base_threshold=th, min_threshold=th, hold_frames=1 + hold, cooldown_frames=cd)
              for th, hold, cd in itertools.product((0.6, 0.7, 0.8, 0.9, 0.95, 0.98), (0, 1, 2, 3), (5, 9))]
    adaptive = [dict(margin_weight=mw, stability_weight=sw, max_threshold=mx)
                for mw, sw, mx in itertools.product((0.1, 0.2, 0.3), (0.1, 0.2), (0.95, 0.99))]
    # quan sát thêm 2/4 bước (~333/667 ms)
    abstain = [dict(clear_threshold=cl, abstain_wait_frames=w, conflict_floor=cf, margin_threshold=mg)
               for cl, w, cf, mg in itertools.product((0.9, 0.95), (2, 4), (0.3, 0.5), (0.1, 0.25))]
    out = {"fixed": [], "adaptive": [], "fixed_abstention": [], "adaptive_abstention": []}
    for s in shared:
        out["fixed"].append({"policy": "fixed", **common, **s})
        out["adaptive"] += [{"policy": "adaptive", **common, **s, **a} for a in adaptive if a["max_threshold"] >= s["base_threshold"]]
        out["fixed_abstention"] += [{"policy": "fixed_abstention", **common, **s, **b} for b in abstain]
        if top_adaptive is not None:
            out["adaptive_abstention"] += [{"policy": "adaptive_abstention", **common, **s, **a, **b}
                                           for a in top_adaptive for b in abstain if a["max_threshold"] >= s["base_threshold"]]
    return out


def app_defaults(step_ms: float, length: int) -> list[dict]:
    """Tham số app đang dùng (config.yaml) quy ra khung: mốc "chưa chỉnh"."""
    common = dict(reverse_block_frames=frames(2500, step_ms), repeat_block_frames=length, uncertain_floor=0.40,
                  min_threshold=0.70, max_threshold=0.95, margin_weight=0.20, stability_weight=0.15,
                  clear_threshold=0.90, conflict_floor=0.50, conflict_frames=frames(700, step_ms) + 1,
                  abstain_wait_frames=frames(600, step_ms), margin_threshold=0.25)
    research = dict(common, base_threshold=0.70, hold_frames=1 + frames(400, step_ms), cooldown_frames=frames(800, step_ms))
    rows = [{"policy": p, "setting": "app C1-C4 (conf 0.7, giữ 400 ms)", **research} for p in POLICY_FLAGS]
    demo = dict(common, base_threshold=0.70, hold_frames=1, cooldown_frames=frames(800, step_ms))
    rows.append({"policy": "fixed", "setting": "app swipe (demo: giữ 0 ms)", **demo})
    return rows


def select(grid_df: pd.DataFrame, budgets: list[float], latency_caps: list[float], cap_budget: float) -> pd.DataFrame:
    """Chọn trên validation. (a) mỗi ngân sách kích hoạt nhầm; (b) ngân sách cap_budget + độ trễ trung vị <= mỗi mức
    (so sánh policy ở CÙNG độ trễ: giữ lâu cũng là "chờ thêm bằng chứng", nên không so ở cùng độ trễ thì không công bằng)."""
    picks = []
    for policy, group in grid_df.groupby("policy", sort=False):
        targets = [(b, None) for b in budgets] + [(cap_budget, cap) for cap in latency_caps]
        for budget, cap in targets:
            ok = group[group.non_command_trigger_rate <= budget]
            if cap is not None:
                ok = ok[ok.median_latency_ms <= cap + 1e-6]
            if ok.empty:
                continue
            best = ok.sort_values(["clean_success_rate", "median_latency_ms"], ascending=[False, True]).iloc[0]
            picks.append({**best.to_dict(), "budget": budget, "latency_cap": cap})
    return pd.DataFrame(picks)


def readable(row: dict, step_ms: float) -> dict:
    out = {"hold_ms": round((row["hold_frames"] - 1) * step_ms), "cooldown_ms": round(row["cooldown_frames"] * step_ms),
           "threshold": row["base_threshold"]}
    if row["policy"].startswith("adaptive"):
        out.update(margin_weight=row["margin_weight"], stability_weight=row["stability_weight"], max_threshold=row["max_threshold"])
    if row["policy"].endswith("abstention"):
        out.update(clear_threshold=row["clear_threshold"], abstain_wait_ms=round(row["abstain_wait_frames"] * step_ms),
                   conflict_floor=row["conflict_floor"], margin_threshold=row["margin_threshold"])
    return out


PARAM_KEYS = ["base_threshold", "min_threshold", "max_threshold", "margin_weight", "stability_weight", "margin_threshold",
              "hold_frames", "cooldown_frames", "uncertain_floor", "clear_threshold", "conflict_floor", "conflict_frames",
              "abstain_wait_frames", "reverse_block_frames", "repeat_block_frames"]


def config_of(row) -> dict:
    return {"policy": row["policy"], **{k: (int(row[k]) if k.endswith("_frames") else float(row[k]))
                                        for k in PARAM_KEYS if k in row and pd.notna(row[k])}}


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", default="data/zoo_predictions")
    parser.add_argument("--model-dir", default="models/zoo")
    parser.add_argument("--runs", default="", help="Mặc định: mọi run có cả _validation.csv và _test.csv")
    parser.add_argument("--commands", default="next,previous")
    parser.add_argument("--budgets", default="0.01,0.02,0.03")
    parser.add_argument("--ci-budget", type=float, default=0.02, help="Ngân sách dùng để tính khoảng tin cậy bootstrap")
    parser.add_argument("--latency-caps", default="1000,1167,1333",
                        help="Mức độ trễ trung vị (ms) để so sánh policy ở cùng độ trễ (bội số của 167 ms ở 6 fps)")
    parser.add_argument("--reuse-grid", action="store_true", help="Dùng lại reports/tuning/<run>_val_grid.csv nếu đã có")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--out-dir", default="reports/tuning")
    args = parser.parse_args()

    pred_dir, out_dir = Path(args.pred_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    commands = [c.strip() for c in args.commands.split(",")]
    budgets = [float(b) for b in args.budgets.split(",")]
    caps = [float(c) for c in args.latency_caps.split(",") if c.strip()]
    runs = [r.strip() for r in args.runs.split(",") if r.strip()] or sorted(
        p.name[:-len("_validation.csv")] for p in pred_dir.glob("*_validation.csv") if (pred_dir / p.name.replace("_validation", "_test")).exists())

    def previous(name):  # giữ kết quả của các run khác (chạy nhiều lần, mỗi lần vài run)
        path = out_dir / name
        return [r for r in pd.read_csv(path).to_dict("records") if r["run"] not in runs] if path.exists() else []

    selected_rows, default_rows, ci_rows = previous("selected.csv"), previous("app_defaults.csv"), previous("bootstrap_ci.csv")
    for run in runs:
        meta = json.loads((Path(args.model_dir) / f"{run}.meta.json").read_text(encoding="utf-8"))
        step_ms, length = float(meta["step_ms"]), int(meta["sequence_length"])
        fps = 1000.0 / step_ms
        val, test = load(pred_dir / f"{run}_validation.csv", commands), load(pred_dir / f"{run}_test.csv", commands)
        print(f"\n===== {run}: validation {len(val)} trials, test {len(test)} trials", flush=True)

        grid_path = out_dir / f"{run}_val_grid.csv"
        if args.reuse_grid and grid_path.exists():
            grid_df = pd.read_csv(grid_path)
        else:
            with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(val, fps)) as pool:
                g = grid(step_ms, length)
                first = [c for p in ("fixed", "adaptive", "fixed_abstention") for c in g[p]]
                rows = list(pool.map(_evaluate, first, chunksize=8))
                adaptive_df = pd.DataFrame([r for r in rows if r["policy"] == "adaptive"])
                top = (adaptive_df.assign(score=adaptive_df.clean_success_rate - adaptive_df.non_command_trigger_rate)
                       .sort_values("score", ascending=False)
                       .drop_duplicates(["margin_weight", "stability_weight", "max_threshold"]).head(2))
                top_adaptive = top[["margin_weight", "stability_weight", "max_threshold"]].to_dict("records")
                rows += list(pool.map(_evaluate, grid(step_ms, length, top_adaptive)["adaptive_abstention"], chunksize=8))
            grid_df = pd.DataFrame(rows)
            grid_df.to_csv(grid_path, index=False)
        print(f"  grid: {len(grid_df)} cấu hình", flush=True)

        picks = select(grid_df, budgets, caps, args.ci_budget)
        with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(test, fps)) as pool:
            test_rows = list(pool.map(_evaluate, [config_of(r) for _, r in picks.iterrows()]))
            defaults = app_defaults(step_ms, length)
            default_test = list(pool.map(_evaluate, [{k: v for k, v in d.items() if k != "setting"} for d in defaults]))
            ci_sets = []
            for cap in [None] + caps:
                sub = picks[(picks.budget == args.ci_budget) & (picks.latency_cap.isna() if cap is None else picks.latency_cap == cap)]
                outs = dict(zip(sub.policy, pool.map(_trial_outcomes, [config_of(r) for _, r in sub.iterrows()])))
                ci_sets.append((cap, outs))
        for cap, outs in ci_sets:
            if "fixed" in outs:
                ci_rows.extend({"run": run, "budget": args.ci_budget, "latency_cap": cap, **r} for r in bootstrap(outs, "fixed"))
        pd.DataFrame(ci_rows).to_csv(out_dir / "bootstrap_ci.csv", index=False)
        for (_, pick), tr in zip(picks.iterrows(), test_rows):
            selected_rows.append({"run": run, "policy": pick["policy"], "budget": pick["budget"], "latency_cap": pick["latency_cap"],
                                  **{f"val_{m}": pick[m] for m in METRICS}, **{f"test_{m}": tr[m] for m in METRICS},
                                  **readable(pick.to_dict(), step_ms), "config": json.dumps(config_of(pick))})
        for d, tr in zip(defaults, default_test):
            default_rows.append({"run": run, "setting": d["setting"], "policy": d["policy"], **{f"test_{m}": tr[m] for m in METRICS}})
        pd.DataFrame(selected_rows).to_csv(out_dir / "selected.csv", index=False)
        pd.DataFrame(default_rows).to_csv(out_dir / "app_defaults.csv", index=False)
        show = pd.DataFrame([r for r in selected_rows if r["run"] == run and pd.isna(r["latency_cap"])])
        with pd.option_context("display.width", 250, "display.max_columns", None, "display.float_format", "{:.3f}".format):
            print(show[["policy", "budget", "test_clean_success_rate", "test_wrong_command_rate",
                        "test_non_command_trigger_rate", "test_median_latency_ms", "threshold", "hold_ms"]].to_string(index=False))


if __name__ == "__main__":
    main()
