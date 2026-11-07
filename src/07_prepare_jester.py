"""Convert selected Jester videos into fixed-length MediaPipe landmark sequences."""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from landmark_utils import create_hands
from sequence_utils import pad_or_sample, extract_two_hand_landmarks

TARGET_LABELS = {
    "Swiping Right": "next",
    "Swiping Left": "previous",
    "Zooming In": "zoom_in",
    "Zooming Out": "zoom_out",
    "Doing other things": "none",
}


def read_labels(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as file:
        rows = list(csv.reader(file))
    return [row[0].strip() for row in rows if row and row[0].strip()]


def prepare(args: argparse.Namespace) -> None:
    labels = read_labels(Path(args.labels))
    videos = sorted(
        (path for path in Path(args.video_root).rglob("*") if path.suffix.lower() in {".avi", ".mp4"}),
        key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
    )
    sequences: list[np.ndarray] = []
    encoded_labels: list[str] = []
    counts = {label: 0 for label in TARGET_LABELS.values()}
    with create_hands(static_image_mode=False, max_num_hands=2) as hands:
        for video in videos:
            video_id = int(video.stem) if video.stem.isdigit() else None
            if video_id is None or video_id < 1 or video_id > len(labels):
                continue
            label = TARGET_LABELS.get(labels[video_id - 1])
            if label is None or counts[label] >= args.max_per_label:
                continue
            capture = cv2.VideoCapture(str(video))
            frames: list[np.ndarray] = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frames.append(extract_two_hand_landmarks(result))
            capture.release()
            if frames:
                sequences.append(pad_or_sample(frames, args.sequence_length))
                encoded_labels.append(label)
                counts[label] += 1
                if all(value >= args.max_per_label for value in counts.values()):
                    break
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, X=np.asarray(sequences), labels=np.asarray(encoded_labels))
    print(f"Saved {len(encoded_labels)} sequences to {output}")
    print(counts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", default="data/jester_sequences.npz")
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--max-per-label", type=int, default=1000)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
