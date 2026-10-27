"""Step 6: extract HaGRID landmarks and split by user_id.

Expected manifest columns: image_path, label, user_id.
"""
import argparse
from pathlib import Path
import cv2
import pandas as pd
from landmark_utils import create_hands, extract_landmarks, FEATURE_NAMES


def extract(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(args.manifest)
    required = {"image_path", "label", "user_id"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Manifest cần các cột: {sorted(required)}")
    rows = []
    with create_hands(static_image_mode=True) as hands:
        for item in manifest.itertuples(index=False):
            image = cv2.imread(str(item.image_path))
            if image is None:
                continue
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            features = extract_landmarks(results)
            if features is not None:
                rows.append(dict(zip(FEATURE_NAMES, features)) | {"label": item.label, "user_id": item.user_id})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Extracted {len(rows)} samples to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="data/hagrid_landmarks.csv")
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()
    if args.extract:
        extract(args)
    data = pd.read_csv(args.output)
    users = data["user_id"].drop_duplicates().tolist()
    split = max(1, int(len(users) * 0.2))
    test_users = set(users[-split:])
    data[data.user_id.isin(test_users)].to_csv("data/hagrid_test.csv", index=False)
    data[~data.user_id.isin(test_users)].to_csv("data/hagrid_train.csv", index=False)
    print(f"Train users: {len(users) - len(test_users)}, test users: {len(test_users)}")


if __name__ == "__main__":
    main()
