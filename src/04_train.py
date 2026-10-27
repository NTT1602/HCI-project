"""Step 4: train and evaluate a Random Forest classifier."""
import argparse
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from landmark_utils import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/landmarks.csv")
    parser.add_argument("--model", default="models/rf.pkl")
    args = parser.parse_args()
    _, X, y = load_dataset(args.data)
    if len(set(y)) < 2:
        raise ValueError("Cần ít nhất 2 nhãn khác nhau để train")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced_subsample", n_jobs=-1)
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, predicted):.3f}")
    print(classification_report(y_test, predicted, zero_division=0))
    labels = sorted(set(y))
    sns.heatmap(confusion_matrix(y_test, predicted, labels=labels), annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(args.model).with_suffix(".png"), dpi=150)
    joblib.dump(model, args.model)
    print(f"Saved model: {args.model}")


if __name__ == "__main__":
    main()
