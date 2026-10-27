"""Step 5: predict gestures and show label/probability."""
import argparse
import cv2
import joblib
from landmark_utils import create_hands, draw_landmarks, process_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/rf.pkl")
    args = parser.parse_args()
    model = joblib.load(args.model)
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
                text = "No hand"
                if features is not None:
                    probabilities = model.predict_proba([features])[0]
                    index = probabilities.argmax()
                    text = f"{model.classes_[index]} ({probabilities[index]:.0%})"
                cv2.putText(frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Step 5 - Realtime", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            camera.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
