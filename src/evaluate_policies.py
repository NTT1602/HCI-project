"""Replay one prediction log through C1-C4 decision policies.

Example:
    python src/evaluate_policies.py --input data/predictions.csv --fps 8 --hold-frames 3 --cooldown-frames 8

The evaluator never reruns the classifier. It reuses identical predictions and
changes only the post-model policy, which makes the comparison reproducible.

Prediction log columns: frame_idx, trial_id, trial_type (intentional /
background / near_miss), intended_action, predicted_label, confidence, margin
and optionally gesture_start_frame (latency is measured from it) and p_<class>
(xác suất từng lớp; abstention C3/C4 cần p_next/p_previous để dò mâu thuẫn hướng).

--commands next,previous: chỉ coi các lớp này là lệnh (giống app: zoom gộp vào none). Xác suất lớp khác cộng
vào p_none, nhãn/confidence/margin tính lại, video zoom trở thành near_miss (cử chỉ có chủ ý nhưng không phải lệnh).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from policies import DecisionPolicy, make_policies

REQUIRED_COLUMNS = {
    "frame_idx", "trial_id", "trial_type", "intended_action",
    "predicted_label", "confidence", "margin",
}
NO_PREDICTION = {"", "none", "No hand"}


def restrict_commands(data: pd.DataFrame, commands: list[str]) -> pd.DataFrame:
    """Chỉ giữ `commands` là lệnh; mọi lớp khác gộp vào none (như app gộp zoom vào none)."""
    prob_cols = [c for c in data.columns if c.startswith("p_")]
    if not prob_cols:
        raise ValueError("--commands cần các cột p_<lớp> trong log dự đoán")
    data = data.copy()
    keep = [f"p_{c}" for c in commands]
    others = [c for c in prob_cols if c not in keep and c != "p_none"]
    data["p_none"] = (data["p_none"] if "p_none" in data else 0.0) + data[others].sum(axis=1)
    data = data.drop(columns=others)
    names = commands + ["none"]
    probs = data[[f"p_{c}" for c in names]].to_numpy()
    order = np.sort(probs, axis=1)
    data["predicted_label"] = np.array(names)[probs.argmax(axis=1)]
    data["confidence"] = order[:, -1].round(5)
    data["margin"] = (order[:, -1] - order[:, -2]).round(5)
    non_command = (data["trial_type"] == "intentional") & ~data["intended_action"].isin(commands)
    data.loc[non_command, "trial_type"] = "near_miss"
    data.loc[non_command, "intended_action"] = "none"
    return data


@dataclass
class TrialArrays:
    """Một trial đã tách sẵn thành mảng (chuẩn bị một lần, chạy qua nhiều policy/tham số)."""

    trial_id: str
    trial_type: str
    intended: str
    start_frame: int
    first_frame: int
    last_frame: int
    frame_ids: np.ndarray
    labels: np.ndarray
    confidences: np.ndarray
    margins: np.ndarray
    probs: list | None


def prepare_trial(trial: pd.DataFrame) -> TrialArrays:
    trial = trial.sort_values("frame_idx")
    prob_cols = [c for c in trial.columns if c.startswith("p_")]
    names = [c[2:] for c in prob_cols]
    probs = [dict(zip(names, row)) for row in trial[prob_cols].to_numpy(dtype=float)] if prob_cols else None
    first = trial.iloc[0]
    start_frame = int(trial.frame_idx.min())
    if "gesture_start_frame" in trial.columns and pd.notna(first.gesture_start_frame):
        start_frame = int(first.gesture_start_frame)
    return TrialArrays(
        str(first.trial_id), str(first.trial_type), str(first.intended_action), start_frame,
        int(trial.frame_idx.min()), int(trial.frame_idx.max()), trial["frame_idx"].to_numpy(),
        trial["predicted_label"].to_numpy(), trial["confidence"].to_numpy(dtype=float),
        trial["margin"].to_numpy(dtype=float), probs,
    )


def evaluate_trial(trial: pd.DataFrame, policy: DecisionPolicy, fps: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return evaluate_prepared(prepare_trial(trial), policy, fps)


def evaluate_prepared(t: TrialArrays, policy: DecisionPolicy, fps: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    policy.reset()
    events: list[dict[str, Any]] = []
    abstain_frames = 0
    candidate_frames = 0
    for i in range(len(t.labels)):
        label, confidence = t.labels[i], t.confidences[i]
        if label not in NO_PREDICTION and confidence > 0:
            candidate_frames += 1
        decision = policy.update(label, confidence, t.margins[i], t.probs[i] if t.probs is not None else None)
        if decision.status.startswith("abstain"):
            abstain_frames += 1
        if decision.event:
            events.append({
                "trial_id": t.trial_id,
                "frame_idx": int(t.frame_ids[i]),
                "event": decision.event,
                "confidence": float(confidence),
                "threshold": decision.threshold,
                "stability": decision.stability,
            })

    trial_type, intended, start_frame = t.trial_type, t.intended, t.start_frame
    matching = [event for event in events if event["event"] == intended]
    if trial_type == "intentional":
        success = bool(matching)
        miss = not success
        false_events = len(events) - len(matching[:1])
        wrong_command = any(event["event"] != intended for event in events)
        clean = success and not wrong_command
        latency_ms = ((matching[0]["frame_idx"] - start_frame) / fps * 1000) if matching else None
    else:
        success = False
        miss = False
        false_events = len(events)
        wrong_command = False
        clean = False
        latency_ms = None

    duration_seconds = max(1.0 / fps, (t.last_frame - t.first_frame + 1) / fps)
    result = {
        "trial_id": t.trial_id,
        "trial_type": trial_type,
        "intended_action": intended,
        "events": len(events),
        "successful_command": int(success),
        "clean_success": int(clean),
        "command_miss": int(miss),
        "wrong_command": int(wrong_command),
        "false_trigger_events": false_events,
        "duration_seconds": duration_seconds,
        "latency_ms": latency_ms,
        "abstention_rate": abstain_frames / candidate_frames if candidate_frames else 0.0,
    }
    for event in events:
        event["policy_trial_type"] = trial_type
    return result, events


def _trigger_rate(group: pd.DataFrame, trial_type: str) -> float | None:
    subset = group[group.trial_type == trial_type]
    return float((subset.events > 0).mean()) if len(subset) else None


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy_name, group in results.groupby("policy", sort=False):
        intentional = group[group.trial_type == "intentional"]
        non_command = group[group.trial_type != "intentional"]
        total_minutes = group.duration_seconds.sum() / 60
        latency = intentional.latency_ms.dropna() if len(intentional) else pd.Series(dtype=float)
        rows.append({
            "policy": policy_name,
            "trials": len(group),
            "intentional_trials": len(intentional),
            "command_success_rate": intentional.successful_command.mean() if len(intentional) else 0.0,
            "clean_success_rate": intentional.clean_success.mean() if len(intentional) else 0.0,
            "command_miss_rate": intentional.command_miss.mean() if len(intentional) else 0.0,
            "wrong_command_rate": intentional.wrong_command.mean() if len(intentional) else 0.0,
            "false_trigger_events": int(group.false_trigger_events.sum()),
            "false_triggers_per_minute": group.false_trigger_events.sum() / total_minutes if total_minutes else 0.0,
            "background_trigger_rate": _trigger_rate(group, "background"),
            "near_miss_trigger_rate": _trigger_rate(group, "near_miss"),
            "non_command_trigger_rate": float((non_command.events > 0).mean()) if len(non_command) else None,
            "mean_latency_ms": latency.mean() if len(latency) else None,
            "median_latency_ms": latency.median() if len(latency) else None,
            "abstention_rate": group.abstention_rate.mean() if len(group) else 0.0,
            "background_trials_with_trigger": int((non_command.events > 0).sum()),
        })
    return pd.DataFrame(rows)


def run_policies(data: pd.DataFrame, policies: dict[str, DecisionPolicy], fps: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    trial_results: list[dict[str, Any]] = []
    event_results: list[dict[str, Any]] = []
    trials = [prepare_trial(trial) for _, trial in data.groupby("trial_id", sort=False)]
    for policy_name, policy in policies.items():
        for trial in trials:
            result, events = evaluate_prepared(trial, policy, fps)
            result["policy"] = policy_name
            trial_results.append(result)
            for event in events:
                event["policy"] = policy_name
                event_results.append(event)
    return pd.DataFrame(trial_results), pd.DataFrame(event_results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/predictions.csv")
    parser.add_argument("--output-dir", default="reports/evaluation")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--min-threshold", type=float, default=0.70)
    parser.add_argument("--margin-threshold", type=float, default=0.10)
    parser.add_argument("--hold-frames", type=int, default=5)
    parser.add_argument("--cooldown-frames", type=int, default=20)
    parser.add_argument("--sweep", default="",
                        help="Danh sách ngưỡng cách nhau dấu phẩy, vd 0.5,0.6,0.7,0.8,0.9 (đường trade-off RQ1)")
    parser.add_argument("--commands", default="",
                        help="Chỉ các lớp này là lệnh, vd next,previous (giống app). Trống = mọi lớp trừ none")
    parser.add_argument("--max-threshold", type=float, default=0.95)
    parser.add_argument("--margin-weight", type=float, default=0.20)
    parser.add_argument("--stability-weight", type=float, default=0.15)
    parser.add_argument("--uncertain-floor", type=float, default=0.40)
    parser.add_argument("--clear-threshold", type=float, default=0.90)
    parser.add_argument("--conflict-floor", type=float, default=0.50)
    parser.add_argument("--conflict-frames", type=int, default=4)
    parser.add_argument("--abstain-wait-frames", type=int, default=4)
    parser.add_argument("--reverse-block-frames", type=int, default=0, help="0 = tắt; app dùng 2.5 s")
    parser.add_argument("--repeat-block-frames", type=int, default=0,
                        help="0 = tắt; = độ dài cửa sổ model để mô phỏng app reset GRU sau mỗi lệnh")
    args = parser.parse_args()

    data = pd.read_csv(args.input, keep_default_na=False)
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Prediction log thiếu cột: {sorted(missing)}")
    if data.empty:
        raise ValueError("Prediction log đang rỗng")
    data["confidence"] = pd.to_numeric(data["confidence"])
    data["margin"] = pd.to_numeric(data["margin"])
    if args.commands:
        data = restrict_commands(data, [c.strip() for c in args.commands.split(",") if c.strip()])
    if "gesture_start_frame" in data.columns:
        data["gesture_start_frame"] = pd.to_numeric(data["gesture_start_frame"], errors="coerce")

    def policies_at(threshold: float) -> dict[str, DecisionPolicy]:
        return make_policies(
            base_threshold=threshold,
            min_threshold=min(args.min_threshold, threshold),
            max_threshold=args.max_threshold,
            margin_weight=args.margin_weight,
            stability_weight=args.stability_weight,
            margin_threshold=args.margin_threshold,
            hold_frames=args.hold_frames,
            cooldown_frames=args.cooldown_frames,
            uncertain_floor=args.uncertain_floor,
            clear_threshold=args.clear_threshold,
            conflict_floor=args.conflict_floor,
            conflict_frames=args.conflict_frames,
            abstain_wait_frames=args.abstain_wait_frames,
            reverse_block_frames=args.reverse_block_frames,
            repeat_block_frames=args.repeat_block_frames,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_frame, event_frame = run_policies(data, policies_at(args.threshold), args.fps)
    summary = summarize(trial_frame)
    trial_frame.to_csv(output_dir / "trial_results.csv", index=False)
    summary.to_csv(output_dir / "policy_summary.csv", index=False)
    event_frame.to_csv(output_dir / "policy_events.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.float_format", "{:.3f}".format):
        print(f"Threshold = {args.threshold}")
        print(summary.to_string(index=False))

    if args.sweep:
        sweeps = []
        for threshold in [float(value) for value in args.sweep.split(",") if value.strip()]:
            trial_frame, _ = run_policies(data, policies_at(threshold), args.fps)
            sweeps.append(summarize(trial_frame).assign(threshold=threshold))
            print(f"  sweep threshold={threshold} done")
        sweep = pd.concat(sweeps, ignore_index=True)
        columns = ["threshold", "policy", "command_success_rate", "false_triggers_per_minute",
                   "background_trigger_rate", "near_miss_trigger_rate", "median_latency_ms", "abstention_rate"]
        sweep.to_csv(output_dir / "threshold_sweep.csv", index=False)
        with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
            print(sweep[columns].to_string(index=False))
    print(f"\nSaved evaluation files to: {output_dir}")


if __name__ == "__main__":
    main()
