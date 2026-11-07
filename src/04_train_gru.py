"""Train a GRU classifier on fixed-length two-hand landmark sequences."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/jester_sequences.npz")
    parser.add_argument("--model", default="models/gru.keras")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    import tensorflow as tf

    data = np.load(args.data)
    X = data["X"].astype("float32")
    encoder = LabelEncoder()
    y = encoder.fit_transform(data["labels"].astype(str))
    if len(encoder.classes_) < 2:
        raise ValueError("Cần ít nhất 2 nhãn khác nhau để train")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=X.shape[1:]),
            tf.keras.layers.GRU(96, return_sequences=True),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.GRU(48),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(len(encoder.classes_), activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
    )
    print(f"Test accuracy: {model.evaluate(X_test, y_test, verbose=0)[1]:.3f}")
    target = Path(args.model)
    target.parent.mkdir(parents=True, exist_ok=True)
    model.save(target)
    target.with_suffix(".labels.json").write_text(
        json.dumps(encoder.classes_.tolist(), ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved model: {target}")


if __name__ == "__main__":
    main()
