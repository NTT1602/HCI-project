"""Record frame-level model predictions for offline policy comparison.

Run one trial at a time and pass the same metadata for every frame. Press q to
stop. The resulting CSV can be replayed by evaluate_policies.py.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import cv2
import joblib

from landmark_utils import create_hands, draw_landmarks, process_frame

FIELDS = [
    "timestamp", "frame_idx", "participant_id", "session_id", "trial_id",
    "trial_type", "intended_action", "predicted_label", "confidence",
    "second_confidence", "margin",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/rf.pkl")
    parser.add_argument("--output", default="data/predictions.csv")
    parser.add_argument("--participant-id", default="local")
    parser.add_argument("--session-id", default="session-1")
    parser.add_argument("--trial-id", required=True, help="Unique id for this recording trial")
    parser.add_argument("--trial-type", choices=["intentional", "near_miss", "background"], required=True)
    parser.add_argument("--intended-action", default="none", help="Expected label, or none for non-command trials")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    new_file = not output.exists()
    model = joblib.load(args.model)
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Không mở được webcam")

    with output.open("a", newline="", encoding="utf-8") as file, create_hands() as hands:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        frame_idx = 0
        try:
            while True:
                ok, frame = camera.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                results, features = process_frame(frame, hands)
                draw_landmarks(frame, results)
                label, confidence, second_confidence = "none", 0.0, 0.0
                if features is not None:
                    probabilities = model.predict_proba([features])[0]
                    ranking = probabilities.argsort()[::-1]
                    label = str(model.classes_[ranking[0]])
                    confidence = float(probabilities[ranking[0]])
                    if len(ranking) > 1:
                        second_confidence = float(probabilities[ranking[1]])
                writer.writerow({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "frame_idx": frame_idx,
                    "participant_id": args.participant_id,
                    "session_id": args.session_id,
                    "trial_id": args.trial_id,
                    "trial_type": args.trial_type,
                    "intended_action": args.intended_action,
                    "predicted_label": label,
                    "confidence": f"{confidence:.6f}",
                    "second_confidence": f"{second_confidence:.6f}",
                    "margin": f"{confidence - second_confidence:.6f}",
                })
                file.flush()
                cv2.putText(frame, f"{label} {confidence:.0%} | q=quit", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Prediction logger", frame)
                frame_idx += 1
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            camera.release()
            cv2.destroyAllWindows()

    print(f"Prediction log appended: {output}")


if __name__ == "__main__":
    main()
