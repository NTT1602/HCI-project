"""Step 3: collect labeled landmarks from webcam or video files."""
import argparse
from pathlib import Path
import sys
import cv2
from landmark_utils import LABELS, create_hands, draw_landmarks, process_frame, save_sample

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

VALID_LABELS = set(LABELS.values())
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def resolve_label(raw_label: str) -> str:
    """Accept either key ('1', '2', '3', '0') or label name ('next', 'previous', ...)."""
    if raw_label in LABELS:
        return LABELS[raw_label]
    if raw_label in VALID_LABELS:
        return raw_label
    raise ValueError(f"Nhãn không hợp lệ: '{raw_label}'. Chọn một trong: {sorted(VALID_LABELS)}")


def extract_from_video(
    video_path: Path,
    label: str,
    output_path: str,
    user_id: str,
    hands: object,
    flip: bool = True,
) -> tuple[int, int]:
    """Process a single video file and append landmarks to output_path.

    Returns (saved_frames, total_frames).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[CẢNH BÁO] Không mở được video: {video_path}")
        return 0, 0
    saved = 0
    total = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            total += 1
            if flip:
                frame = cv2.flip(frame, 1)
            _, features = process_frame(frame, hands)
            if features is not None:
                save_sample(output_path, features, label, user_id)
                saved += 1
    finally:
        cap.release()
    print(f"  + [{label}] {video_path.name}: lưu {saved}/{total} frames có bàn tay (user_id={user_id})")
    return saved, total


def collect_from_videos(args: argparse.Namespace) -> None:
    tasks: list[tuple[Path, str]] = []

    if args.video_dir:
        root = Path(args.video_dir)
        if not root.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục video: {root}")
        for folder in sorted(root.iterdir()):
            if not folder.is_dir():
                continue
            try:
                label = resolve_label(folder.name)
            except ValueError:
                print(f"[BỎ QUA] Thư mục '{folder.name}' không trùng tên nhãn ({sorted(VALID_LABELS)})")
                continue
            videos = sorted(
                p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
            )
            for video in videos:
                tasks.append((video, label))

    if args.video:
        if not args.label:
            raise ValueError("Cần truyền --label (next, previous, play_pause, none) khi dùng --video")
        label = resolve_label(args.label)
        for v in args.video:
            tasks.append((Path(v), label))

    if not tasks:
        print("Không tìm thấy file video nào để xử lý.")
        return

    if args.overwrite and Path(args.output).exists():
        Path(args.output).unlink()
        print(f"Đã xóa file cũ để ghi mới: {args.output}")

    total_saved = 0
    total_frames = 0
    flip = not args.no_flip
    with create_hands() as hands:
        for video_path, label in tasks:
            sample_user = args.user_id if args.user_id != "auto" else video_path.stem
            saved, frames = extract_from_video(
                video_path, label, args.output, sample_user, hands, flip=flip
            )
            total_saved += saved
            total_frames += frames

    print(
        f"Hoàn tất! Đã xử lý {len(tasks)} video, lưu {total_saved}/{total_frames} mẫu vào {args.output}"
    )


def collect_from_webcam(args: argparse.Namespace) -> None:
    if args.overwrite and Path(args.output).exists():
        Path(args.output).unlink()
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
                if not args.no_flip:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/landmarks.csv")
    parser.add_argument("--user-id", default="local", help="ID người dùng, hoặc 'auto' để lấy theo tên video")
    parser.add_argument("--video", nargs="+", help="Một hoặc nhiều đường dẫn file video của cùng 1 nhãn")
    parser.add_argument("--label", help="Nhãn cho --video: next, previous, play_pause, none (hoặc 1, 2, 3, 0)")
    parser.add_argument(
        "--video-dir",
        help="Thư mục chứa các thư mục con theo tên nhãn (next/, previous/, play_pause/, none/)",
    )
    parser.add_argument("--no-flip", action="store_true", help="Không lật gương frame (nếu video đã lật sẵn)")
    parser.add_argument("--overwrite", action="store_true", help="Xóa file CSV cũ trước khi trích xuất")
    args = parser.parse_args()

    if args.video or args.video_dir:
        collect_from_videos(args)
    else:
        collect_from_webcam(args)


if __name__ == "__main__":
    main()
