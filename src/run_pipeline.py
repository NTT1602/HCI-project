"""Run the whole offline pipeline: train GRU -> simulate streams -> compare C1-C4.

Example:
    python src/run_pipeline.py                 # full run
    python src/run_pipeline.py --skip-train    # reuse models/gru.keras
    python src/run_pipeline.py --quick         # small smoke run (few epochs, 100 trials/label)

Time unit: one sequence step. With the default 16-step gesture (~2 s of a
Jester clip) the simulated stream runs at ~8 fps, so policy parameters are
given in steps: hold 3 frames (~0.4 s), cooldown 8 frames (~1 s).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent


def run(script: str, *args: str) -> None:
    command = [sys.executable, str(SRC / script), *args]
    print("\n>>> " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/jester_landmarks.csv")
    parser.add_argument("--model", default="models/gru.keras")
    parser.add_argument("--predictions", default="data/predictions.csv")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--fps", default="8")
    parser.add_argument("--threshold", default="0.80")
    parser.add_argument("--hold-frames", default="3")
    parser.add_argument("--cooldown-frames", default="8")
    parser.add_argument("--margin-threshold", default="0.20")
    parser.add_argument("--sweep", default="0.5,0.6,0.7,0.8,0.85,0.9,0.95")
    args = parser.parse_args()

    if not args.skip_train:
        run("04_train_gru.py", "--data", args.data, "--model", args.model,
            "--report-dir", args.report_dir, *(["--epochs", "3"] if args.quick else []))
    run("06_simulate_stream.py", "--data", args.data, "--model", args.model,
        "--output", args.predictions, *(["--max-per-label", "100"] if args.quick else []))
    run("evaluate_policies.py", "--input", args.predictions,
        "--output-dir", str(Path(args.report_dir) / "evaluation"),
        "--fps", args.fps, "--threshold", args.threshold,
        "--hold-frames", args.hold_frames, "--cooldown-frames", args.cooldown_frames,
        "--margin-threshold", args.margin_threshold, "--sweep", args.sweep)


if __name__ == "__main__":
    main()
