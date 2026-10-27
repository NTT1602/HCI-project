"""Step 6: extract HaGRID landmarks and split by user_id.

Expected manifest columns: image_path, label, user_id.
"""
import argparse
from pathlib import Path
import cv2
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
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
    if len(users) < 2:
        raise ValueError("HaGRID cần ít nhất 2 user_id để split theo người dùng")
    split = max(1, int(len(users) * 0.2))
    test_users = set(users[-split:])
    test_data = data[data.user_id.isin(test_users)]
    train_data = data[~data.user_id.isin(test_users)]
    output_dir = Path(args.output).parent
    train_data.to_csv(output_dir / "hagrid_train.csv", index=False)
    test_data.to_csv(output_dir / "hagrid_test.csv", index=False)
    model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(train_data[FEATURE_NAMES], train_data["label"])
    predicted = model.predict(test_data[FEATURE_NAMES])
    print(f"HaGRID accuracy (split theo user_id): {accuracy_score(test_data['label'], predicted):.3f}")
    print(classification_report(test_data["label"], predicted, zero_division=0))
    print(f"Train users: {len(users) - len(test_users)}, test users: {len(test_users)}")


if __name__ == "__main__":
    main()
