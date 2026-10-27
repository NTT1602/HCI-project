"""Step 3: collect labeled landmarks. Keys: 1/2/3/0; q quits."""
import argparse
import cv2
from landmark_utils import LABELS, create_hands, draw_landmarks, process_frame, save_sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/landmarks.csv")
    parser.add_argument("--user-id", default="local")
    args = parser.parse_args()
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Không mở được webcam")
    current_label = None
    saved = 0
    with create_hands() as hands:
        try:
            while True:
                ok, frame = camera.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                results, features = process_frame(frame, hands)
                draw_landmarks(frame, results)
                if features is not None and current_label is not None:
                    save_sample(args.output, features, current_label, args.user_id)
                    saved += 1
                text = f"1:next 2:previous 3:play_pause 0:none | active={current_label or '-'} | saved={saved}"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                cv2.imshow("Step 3 - Collect", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if chr(key) in LABELS:
                    current_label = LABELS[chr(key)]
        finally:
            camera.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
