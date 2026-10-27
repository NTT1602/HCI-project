"""Step 1: open a mirrored webcam preview. Press q to quit."""
import cv2


def main() -> None:
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Không mở được webcam. Hãy kiểm tra quyền camera hoặc camera index.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, "Press q to quit", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Step 1 - Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
