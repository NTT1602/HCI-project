"""Realtime inference for the GRU landmark-sequence model.

Examples:
    python src/05_realtime_gru.py --policy adaptive_abstention
    # Record a real trial in the same format as data/predictions.csv:
    python src/05_realtime_gru.py --log data/real_trials.csv --trial-id p01_t03 \
        --trial-type background --intended none

The model was trained on ~8 steps/second (16 steps ~ 2 s gesture), so by
default only every 4th webcam frame (30 fps -> ~7.5 fps) enters the sequence.
"""
import argparse
import csv
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from landmark_utils import create_hands, draw_landmarks
from policies import make_policies
from sequence_utils import add_motion_features, extract_two_hand_landmarks
from gate import gesture_to_slide_action

LOG_HEADER = ["trial_id", "trial_type", "intended_action", "frame_idx", "predicted_label", "confidence", "margin"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/gru.keras")
    parser.add_argument("--frame-step", type=int, default=4, help="Dùng 1 trên N frame webcam")
    parser.add_argument("--policy", default="fixed",
                        choices=["fixed", "adaptive", "fixed_abstention", "adaptive_abstention"])
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--margin-threshold", type=float, default=0.20)
    parser.add_argument("--hold-frames", type=int, default=3)
    parser.add_argument("--cooldown-frames", type=int, default=8)
    parser.add_argument("--log", default=None, help="CSV ghi prediction từng frame (cho evaluate_policies.py)")
    parser.add_argument("--trial-id", default="live")
    parser.add_argument("--trial-type", default="intentional", choices=["intentional", "background", "near_miss"])
    parser.add_argument("--intended", default="none", help="Lệnh dự định: next/previous/zoom_in/zoom_out/none")
    args = parser.parse_args()

    model_path = Path(args.model)
    model = tf.keras.models.load_model(model_path)
    labels = json.loads(model_path.with_suffix(".labels.json").read_text(encoding="utf-8"))
    sequence_length = int(model.input_shape[1])
    policy = make_policies(
        base_threshold=args.threshold,
        margin_threshold=args.margin_threshold,
        hold_frames=args.hold_frames,
        cooldown_frames=args.cooldown_frames,
    )[args.policy]

    log_writer = None
    log_file = None
    if args.log:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not log_path.exists()
        log_file = open(log_path, "a", newline="", encoding="utf-8")
        log_writer = csv.writer(log_file)
        if new_file:
            log_writer.writerow(LOG_HEADER)

    history: deque[np.ndarray] = deque(maxlen=sequence_length)
    previous_features = None
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Không mở được webcam")
    label, confidence, margin, status = "none", 0.0, 0.0, "-"
    camera_frame = 0
    step = 0
    with create_hands(max_num_hands=1) as hands:
        try:
            while True:
                ok, frame = camera.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw_landmarks(frame, result)
                if camera_frame % args.frame_step == 0:
                    current_features = extract_two_hand_landmarks(result)
                    history.append(add_motion_features(current_features, previous_features))
                    previous_features = current_features
                    window = list(history)
                    if len(window) < sequence_length:
                        pad = [np.zeros_like(window[0])] * (sequence_length - len(window))
                        window = pad + window
                    probabilities = model.predict(np.asarray([window]), verbose=0)[0]
                    order = np.argsort(probabilities)
                    label = labels[int(order[-1])]
                    confidence = float(probabilities[order[-1]])
                    margin = float(probabilities[order[-1]] - probabilities[order[-2]])
                    decision = policy.update(label, confidence, margin)
                    status = decision.status
                    if decision.event:
                        print(gesture_to_slide_action(decision.event) or decision.event, flush=True)
                    if log_writer:
                        log_writer.writerow([args.trial_id, args.trial_type, args.intended, step,
                                             label, f"{confidence:.5f}", f"{margin:.5f}"])
                    step += 1
                camera_frame += 1
                cv2.putText(frame, f"{label} {confidence:.0%} [{status}]", (20, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("GRU realtime", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            camera.release()
            cv2.destroyAllWindows()
            if log_file:
                log_file.close()


if __name__ == "__main__":
    main()
