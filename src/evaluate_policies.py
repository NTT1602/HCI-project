"""Replay one prediction log through C1-C4 decision policies.

Example:
    python src/evaluate_policies.py --input data/predictions.csv

The evaluator never reruns the classifier. It reuses identical predictions and
changes only the post-model policy, which makes the comparison reproducible.
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


def evaluate_trial(trial: pd.DataFrame, policy: DecisionPolicy, fps: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    policy.reset()
    events: list[dict[str, Any]] = []
    abstain_frames = 0
    candidate_frames = 0
    for row in trial.sort_values("frame_idx").itertuples(index=False):
        if row.predicted_label not in {"", "none", "No hand"} and row.confidence > 0:
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

    trial_type = str(trial.iloc[0].trial_type)
    intended = str(trial.iloc[0].intended_action)
    matching = [event for event in events if event["event"] == intended]
    if trial_type == "intentional":
        success = bool(matching)
        miss = not success
        false_events = len(events) - len(matching[:1])
        latency_ms = ((matching[0]["frame_idx"] - int(trial.frame_idx.min())) / fps * 1000) if matching else None
    else:
        success = False
        miss = False
        false_events = len(events)
        latency_ms = None

    duration_seconds = max(1.0 / fps, (int(trial.frame_idx.max()) - int(trial.frame_idx.min()) + 1) / fps)
    result = {
        "trial_id": str(trial.iloc[0].trial_id),
        "trial_type": trial_type,
        "intended_action": intended,
        "events": len(events),
        "successful_command": int(success),
        "command_miss": int(miss),
        "false_trigger_events": false_events,
        "duration_seconds": duration_seconds,
        "latency_ms": latency_ms,
        "abstention_rate": abstain_frames / candidate_frames if candidate_frames else 0.0,
    }
    for event in events:
        event["policy_trial_type"] = trial_type
    return result, events


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy_name, group in results.groupby("policy"):
        intentional = group[group.trial_type == "intentional"]
        non_command = group[group.trial_type != "intentional"]
        total_minutes = group.duration_seconds.sum() / 60
        rows.append({
            "policy": policy_name,
            "trials": len(group),
            "intentional_trials": len(intentional),
            "command_success_rate": intentional.successful_command.mean() if len(intentional) else 0.0,
            "command_miss_rate": intentional.command_miss.mean() if len(intentional) else 0.0,
            "false_trigger_events": int(group.false_trigger_events.sum()),
            "false_triggers_per_minute": group.false_trigger_events.sum() / total_minutes if total_minutes else 0.0,
            "mean_latency_ms": intentional.latency_ms.dropna().mean() if intentional.latency_ms.notna().any() else None,
            "abstention_rate": group.abstention_rate.mean() if len(group) else 0.0,
            "background_trials_with_trigger": int((non_command.events > 0).sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/predictions.csv")
    parser.add_argument("--output-dir", default="data/evaluation")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--margin-threshold", type=float, default=0.10)
    parser.add_argument("--hold-frames", type=int, default=5)
    parser.add_argument("--cooldown-frames", type=int, default=20)
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Prediction log thiếu cột: {sorted(missing)}")
    if data.empty:
        raise ValueError("Prediction log đang rỗng")
    data["confidence"] = pd.to_numeric(data["confidence"])
    data["margin"] = pd.to_numeric(data["margin"])
    policies = make_policies(
        base_threshold=args.threshold,
        margin_threshold=args.margin_threshold,
        hold_frames=args.hold_frames,
        cooldown_frames=args.cooldown_frames,
    )

    trial_results: list[dict[str, Any]] = []
    event_results: list[dict[str, Any]] = []
    for policy_name, policy in policies.items():
        for _, trial in data.groupby("trial_id", sort=False):
            result, events = evaluate_trial(trial, policy, args.fps)
            result["policy"] = policy_name
            trial_results.append(result)
            for event in events:
                event["policy"] = policy_name
                event_results.append(event)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_frame = pd.DataFrame(trial_results)
    summary = summarize(trial_frame)
    trial_frame.to_csv(output_dir / "trial_results.csv", index=False)
    summary.to_csv(output_dir / "policy_summary.csv", index=False)
    pd.DataFrame(event_results).to_csv(output_dir / "policy_events.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved evaluation files to: {output_dir}")


if __name__ == "__main__":
    main()
