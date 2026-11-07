"""Realtime inference for the GRU landmark-sequence model."""
import argparse
import json
from collections import deque

import cv2
import numpy as np
import tensorflow as tf

from landmark_utils import create_hands, draw_landmarks
from sequence_utils import extract_two_hand_landmarks
from gate import GestureGate, gesture_to_slide_action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/gru.keras")
    parser.add_argument("--sequence-length", type=int, default=32)
    args = parser.parse_args()
    model = tf.keras.models.load_model(args.model)
    labels = json.loads(open(args.model.replace(".keras", ".labels.json"), encoding="utf-8").read())
    history: deque[np.ndarray] = deque(maxlen=args.sequence_length)
    gate = GestureGate()
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Không mở được webcam")
    with create_hands(max_num_hands=2) as hands:
        try:
            while True:
                ok, frame = camera.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw_landmarks(frame, result)
                history.append(extract_two_hand_landmarks(result))
                label, confidence = "none", 0.0
                if len(history) == args.sequence_length:
                    probabilities = model.predict(np.asarray([history]), verbose=0)[0]
                    index = int(np.argmax(probabilities))
                    label, confidence = labels[index], float(probabilities[index])
                event = gate.update(label, confidence)
                if event:
                    print(gesture_to_slide_action(event) or event)
                cv2.putText(frame, f"{label} {confidence:.0%}", (20, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("GRU realtime", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            camera.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
