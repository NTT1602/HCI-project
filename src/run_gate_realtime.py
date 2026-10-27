"""Optional realtime demo combining model prediction and Step 7 gate."""
import argparse
import cv2
import joblib
from landmark_utils import create_hands, draw_landmarks, process_frame
from gate import GestureGate, gesture_to_slide_action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/rf.pkl")
    args = parser.parse_args()
    model = joblib.load(args.model)
    gate = GestureGate()
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Không mở được webcam")
    with create_hands() as hands:
        try:
            while True:
                ok, frame = camera.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                results, features = process_frame(frame, hands)
                draw_landmarks(frame, results)
                label, confidence = "none", 0.0
                if features is not None:
                    probabilities = model.predict_proba([features])[0]
                    index = probabilities.argmax()
                    label, confidence = str(model.classes_[index]), float(probabilities[index])
                event = gate.update(label, confidence)
                if event:
                    print(gesture_to_slide_action(event) or event)
                cv2.putText(frame, f"{label} {confidence:.0%}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Step 7 - Gesture gate", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            camera.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
