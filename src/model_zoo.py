"""Các kiến trúc phân loại chuỗi để so sánh với GRU gốc.

Mọi model nhận cùng đầu vào (T bước x 70 đặc trưng: 63 landmark thô + 7 motion, xem sequence_utils.py) và
trả về softmax trên CLASSES. Chỉ dùng layer có sẵn của Keras (không custom layer) để app
HCI-Slideshow-Interaction nạp được bằng `tf.keras.models.load_model` như gru.keras hiện tại.

  gru        GRU 96 -> GRU 48 (kiến trúc gốc của 04_train_gru.py)
  gru_small  GRU 64 (nhẹ hơn, nhanh hơn)
  lstm       LSTM 96 -> LSTM 48
  bigru      BiGRU 64 -> BiGRU 32 (hai chiều trong cửa sổ; vẫn chạy được theo luồng vì mỗi lần dự đoán nhận cả cửa sổ)
  tcn        Conv1D giãn (dilation 1-2-4, causal) -> pooling
  attention  Conv1D -> 2 khối self-attention (không positional encoding: hướng chuyển động đã có trong dx/dy từng bước)
  mlp        Flatten -> Dense (không mô hình hóa thời gian; mốc để xem cấu trúc chuỗi có giúp không)
"""
from __future__ import annotations


def _normalizer(tf, train_x):
    norm = tf.keras.layers.Normalization(axis=-1)
    norm.adapt(train_x.reshape(-1, train_x.shape[-1]))
    return norm


def _head(tf, x, n_classes, dropout=0.25):
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    return tf.keras.layers.Dense(n_classes, activation="softmax")(x)


def build(tf, name: str, train_x, n_classes: int):
    inp = tf.keras.layers.Input(shape=train_x.shape[1:])
    x = _normalizer(tf, train_x)(inp)
    L = tf.keras.layers
    if name == "gru":
        x = L.GRU(96, return_sequences=True)(x)
        x = L.Dropout(0.25)(x)
        x = L.GRU(48)(x)
    elif name == "gru_small":
        x = L.GRU(64)(x)
    elif name == "lstm":
        x = L.LSTM(96, return_sequences=True)(x)
        x = L.Dropout(0.25)(x)
        x = L.LSTM(48)(x)
    elif name == "bigru":
        x = L.Bidirectional(L.GRU(64, return_sequences=True))(x)
        x = L.Dropout(0.25)(x)
        x = L.Bidirectional(L.GRU(32))(x)
    elif name == "tcn":
        for rate in (1, 2, 4):
            y = L.Conv1D(64, 3, padding="causal", dilation_rate=rate, activation="relu")(x)
            y = L.Dropout(0.1)(y)
            x = y if x.shape[-1] != 64 else L.Add()([x, y])
        x = L.Concatenate()([L.GlobalAveragePooling1D()(x), L.GlobalMaxPooling1D()(x)])
    elif name == "attention":
        x = L.Conv1D(64, 3, padding="same", activation="relu")(x)
        for _ in range(2):
            a = L.MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.1)(x, x)
            x = L.LayerNormalization()(L.Add()([x, a]))
            f = L.Dense(128, activation="relu")(x)
            f = L.Dense(64)(f)
            x = L.LayerNormalization()(L.Add()([x, f]))
        x = L.GlobalAveragePooling1D()(x)
    elif name == "mlp":
        x = L.Flatten()(x)
        x = L.Dense(128, activation="relu")(x)
        x = L.Dropout(0.25)(x)
        x = L.Dense(64, activation="relu")(x)
    else:
        raise ValueError(f"Model không hỗ trợ: {name}")
    model = tf.keras.Model(inp, _head(tf, x, n_classes), name=name)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


MODELS = ("gru", "gru_small", "lstm", "bigru", "tcn", "attention", "mlp")
