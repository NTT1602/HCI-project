"""Replay one prediction log through C1-C4 decision policies.

Example:
    python src/evaluate_policies.py --input data/predictions.csv --fps 8 --hold-frames 3 --cooldown-frames 8

The evaluator never reruns the classifier. It reuses identical predictions and
changes only the post-model policy, which makes the comparison reproducible.

Prediction log columns: frame_idx, trial_id, trial_type (intentional /
background / near_miss), intended_action, predicted_label, confidence, margin
and optionally gesture_start_frame (latency is measured from it).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from policies import DecisionPolicy, make_policies

REQUIRED_COLUMNS = {
    "frame_idx", "trial_id", "trial_type", "intended_action",
    "predicted_label", "confidence", "margin",
}
NO_PREDICTION = {"", "none", "No hand"}


def evaluate_trial(trial: pd.DataFrame, policy: DecisionPolicy, fps: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    policy.reset()
    events: list[dict[str, Any]] = []
    abstain_frames = 0
    candidate_frames = 0
    trial = trial.sort_values("frame_idx")
    for row in trial.itertuples(index=False):
        if row.predicted_label not in NO_PREDICTION and row.confidence > 0:
            candidate_frames += 1
        decision = policy.update(row.predicted_label, row.confidence, row.margin)
        if decision.status == "abstain":
            abstain_frames += 1
        if decision.event:
            events.append({
                "trial_id": row.trial_id,
                "frame_idx": int(row.frame_idx),
                "event": decision.event,
                "confidence": float(row.confidence),
                "threshold": decision.threshold,
                "stability": decision.stability,
            })

    first = trial.iloc[0]
    trial_type = str(first.trial_type)
    intended = str(first.intended_action)
    start_frame = int(trial.frame_idx.min())
    if "gesture_start_frame" in trial.columns and pd.notna(first.gesture_start_frame):
        start_frame = int(first.gesture_start_frame)
    matching = [event for event in events if event["event"] == intended]
    if trial_type == "intentional":
        success = bool(matching)
        miss = not success
        false_events = len(events) - len(matching[:1])
        wrong_command = any(event["event"] != intended for event in events)
        latency_ms = ((matching[0]["frame_idx"] - start_frame) / fps * 1000) if matching else None
    else:
        success = False
        miss = False
        false_events = len(events)
        wrong_command = False
        latency_ms = None

    duration_seconds = max(1.0 / fps, (int(trial.frame_idx.max()) - int(trial.frame_idx.min()) + 1) / fps)
    result = {
        "trial_id": str(first.trial_id),
        "trial_type": trial_type,
        "intended_action": intended,
        "events": len(events),
        "successful_command": int(success),
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
            "command_miss_rate": intentional.command_miss.mean() if len(intentional) else 0.0,
            "wrong_command_rate": intentional.wrong_command.mean() if len(intentional) else 0.0,
            "false_trigger_events": int(group.false_trigger_events.sum()),
            "false_triggers_per_minute": group.false_trigger_events.sum() / total_minutes if total_minutes else 0.0,
            "background_trigger_rate": _trigger_rate(group, "background"),
            "near_miss_trigger_rate": _trigger_rate(group, "near_miss"),
            "mean_latency_ms": latency.mean() if len(latency) else None,
            "median_latency_ms": latency.median() if len(latency) else None,
            "abstention_rate": group.abstention_rate.mean() if len(group) else 0.0,
            "background_trials_with_trigger": int((non_command.events > 0).sum()),
        })
    return pd.DataFrame(rows)


def run_policies(data: pd.DataFrame, policies: dict[str, DecisionPolicy], fps: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    trial_results: list[dict[str, Any]] = []
    event_results: list[dict[str, Any]] = []
    trials = [trial for _, trial in data.groupby("trial_id", sort=False)]
    for policy_name, policy in policies.items():
        for trial in trials:
            result, events = evaluate_trial(trial, policy, fps)
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
    args = parser.parse_args()

    data = pd.read_csv(args.input, keep_default_na=False)
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Prediction log thiếu cột: {sorted(missing)}")
    if data.empty:
        raise ValueError("Prediction log đang rỗng")
    data["confidence"] = pd.to_numeric(data["confidence"])
    data["margin"] = pd.to_numeric(data["margin"])
    if "gesture_start_frame" in data.columns:
        data["gesture_start_frame"] = pd.to_numeric(data["gesture_start_frame"], errors="coerce")

    def policies_at(threshold: float) -> dict[str, DecisionPolicy]:
        return make_policies(
            base_threshold=threshold,
            min_threshold=min(args.min_threshold, threshold),
            margin_threshold=args.margin_threshold,
            hold_frames=args.hold_frames,
            cooldown_frames=args.cooldown_frames,
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
