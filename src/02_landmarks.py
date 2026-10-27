"""Step 2: draw MediaPipe's 21 hand landmarks."""
import cv2
from landmark_utils import create_hands, draw_landmarks, process_frame


def main() -> None:
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
                results, _ = process_frame(frame, hands)
                draw_landmarks(frame, results)
                cv2.imshow("Step 2 - Hand landmarks", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            camera.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
