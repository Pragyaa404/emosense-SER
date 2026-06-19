"""
predict.py — Rebuilds the exact model architectures from training,
then loads ONLY the weights from the saved .h5 files. This avoids
all Keras version-compatibility issues with deserializing Lambda
layers from saved configs.
"""

import os
import json
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense,
    Dropout, BatchNormalization, GlobalAveragePooling2D,
    GlobalAveragePooling1D, Reshape, Multiply, Add,
    Activation, Concatenate, Bidirectional, LSTM,
    Lambda, Softmax, MultiHeadAttention, LayerNormalization
)
from tensorflow.keras import backend as K

# ── PATHS ──────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# ── LOAD CONFIG (saved from training notebook) ───────────────────────────────
with open(os.path.join(MODELS_DIR, "config.json")) as f:
    CONFIG = json.load(f)

EMOTIONS     = CONFIG["emotions"]
SAMPLE_RATE  = CONFIG["sample_rate"]
N_MELS       = CONFIG["n_mels"]
MAX_LEN      = CONFIG["max_len"]
N_MFCC       = CONFIG["n_mfcc"]
MAX_LEN_MFCC = CONFIG["max_len_mfcc"]
HOP_LENGTH   = CONFIG["hop_length"]
NUM_CLASSES  = len(EMOTIONS)


# ══════════════════════════════════════════════════════════════════════════
# ARCHITECTURE DEFINITIONS — must exactly match the training notebook
# ══════════════════════════════════════════════════════════════════════════

# ── CBAM blocks (Model 1 — CNN-CBAM) ─────────────────────────────────────────
def channel_attention(input_tensor, reduction=8):
    channels = input_tensor.shape[-1]
    avg = GlobalAveragePooling2D()(input_tensor)
    avg = Reshape((1, 1, channels))(avg)
    avg = Dense(channels // reduction, activation='relu')(avg)
    avg = Dense(channels, activation='sigmoid')(avg)
    mx  = Lambda(lambda x: tf.reduce_max(x, axis=[1, 2], keepdims=True),
                 output_shape=lambda s: (s[0], 1, 1, s[-1]))(input_tensor)
    mx  = Dense(channels // reduction, activation='relu')(mx)
    mx  = Dense(channels, activation='sigmoid')(mx)
    scale = Add()([avg, mx])
    scale = Activation('sigmoid')(scale)
    return Multiply()([input_tensor, scale])


def spatial_attention(input_tensor):
    avg = Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True),
                 output_shape=lambda s: (s[0], s[1], s[2], 1))(input_tensor)
    mx  = Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True),
                 output_shape=lambda s: (s[0], s[1], s[2], 1))(input_tensor)
    concat = Concatenate(axis=-1)([avg, mx])
    scale  = Conv2D(1, (7, 7), padding='same', activation='sigmoid')(concat)
    return Multiply()([input_tensor, scale])


def cbam_block(input_tensor, reduction=8):
    x = channel_attention(input_tensor, reduction)
    x = spatial_attention(x)
    return x


def build_cnn_cbam(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = cbam_block(x, reduction=8)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = cbam_block(x, reduction=8)
    x = MaxPooling2D((2, 2))(x)

    x       = Flatten()(x)
    x       = Dense(256, activation='relu')(x)
    x       = Dropout(0.4)(x)
    x       = Dense(128, activation='relu')(x)
    x       = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)


# ── BiLSTM + Attention (Model 2) ──────────────────────────────────────────────
def attention_block(inputs):
    score    = Dense(1, activation='tanh')(inputs)
    weights  = Softmax(axis=1)(score)
    weighted = Multiply()([inputs, weights])
    context_vector = Lambda(
        lambda x: K.sum(x, axis=1),
        output_shape=lambda s: (s[0], s[2])
    )(weighted)
    return context_vector


def build_bilstm_attention(input_shape, num_classes):
    inp = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), padding='same', activation='relu')(inp)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Dropout(0.3)(x)

    shape      = K.int_shape(x)
    time_steps = shape[1]
    freq_bins  = shape[2]
    channels   = shape[3]
    x = Reshape((time_steps, freq_bins * channels))(x)

    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.2)(x)
    x = attention_block(x)

    x   = Dense(128, activation='relu')(x)
    x   = Dropout(0.3)(x)
    out = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=inp, outputs=out)


# ── EmoFormer (Model 3) ────────────────────────────────────────────────────────
def build_emoformer(input_shape, num_classes):
    inp = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), padding='same', activation='relu')(inp)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    shape = K.int_shape(x)
    x     = Reshape((shape[1], shape[2] * shape[3]))(x)

    attn = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x    = Add()([x, attn])
    x    = LayerNormalization()(x)

    ff = Dense(shape[2] * shape[3], activation='relu')(x)
    x  = Add()([x, ff])
    x  = LayerNormalization()(x)

    x   = GlobalAveragePooling1D()(x)
    x   = Dense(128, activation='relu')(x)
    x   = Dropout(0.3)(x)
    out = Dense(num_classes, activation='softmax')(x)

    return Model(inp, out)


# ══════════════════════════════════════════════════════════════════════════
# BUILD MODELS + LOAD WEIGHTS ONLY
# ══════════════════════════════════════════════════════════════════════════
print("Building model architectures...")
MEL_SHAPE  = (N_MELS, MAX_LEN, 1)        # e.g. (128, 128, 1)
MFCC_SHAPE = (MAX_LEN_MFCC, N_MFCC, 1)   # e.g. (128, 40, 1)

cnn_models = [
    build_cnn_cbam(MEL_SHAPE, NUM_CLASSES),
    build_cnn_cbam(MEL_SHAPE, NUM_CLASSES),
    build_cnn_cbam(MEL_SHAPE, NUM_CLASSES),
]
bilstm_model    = build_bilstm_attention(MFCC_SHAPE, NUM_CLASSES)
emoformer_model = build_emoformer(MFCC_SHAPE, NUM_CLASSES)

print("Loading weights into architectures...")
cnn_models[0].load_weights(os.path.join(MODELS_DIR, "cnn_cbam_1.h5"))
cnn_models[1].load_weights(os.path.join(MODELS_DIR, "cnn_cbam_2.h5"))
cnn_models[2].load_weights(os.path.join(MODELS_DIR, "cnn_cbam_3.h5"))
bilstm_model.load_weights(os.path.join(MODELS_DIR, "bilstm_attention.h5"))
emoformer_model.load_weights(os.path.join(MODELS_DIR, "emoformer.h5"))
print("✅ All 5 models loaded successfully")


# ── FEATURE EXTRACTION (must match training exactly) ─────────────────────────
def extract_mel_spectrogram(signal):
    signal, _ = librosa.effects.trim(signal, top_db=20)
    mel    = librosa.feature.melspectrogram(y=signal, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < MAX_LEN:
        pad    = MAX_LEN - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad)), mode="constant")
    else:
        mel_db = mel_db[:, :MAX_LEN]
    return mel_db


def extract_mfcc_features(signal):
    signal, _ = librosa.effects.trim(signal, top_db=20)
    mfcc = librosa.feature.mfcc(
        y=signal, sr=SAMPLE_RATE, n_mfcc=N_MFCC, hop_length=HOP_LENGTH
    )
    return mfcc.T


def pad_mfcc(mfcc_feat):
    if mfcc_feat.shape[0] < MAX_LEN_MFCC:
        pad = MAX_LEN_MFCC - mfcc_feat.shape[0]
        return np.pad(mfcc_feat, ((0, pad), (0, 0)), mode="constant")
    return mfcc_feat[:MAX_LEN_MFCC]


# ── MAIN PREDICTION FUNCTION ──────────────────────────────────────────────────
def predict_emotion(audio_path: str) -> dict:
    signal, _ = librosa.load(audio_path, sr=SAMPLE_RATE)

    mel_feat = extract_mel_spectrogram(signal)
    mel_feat = (mel_feat - mel_feat.mean()) / (mel_feat.std() + 1e-8)
    mel_input = mel_feat[np.newaxis, ..., np.newaxis]

    cnn_preds = [m.predict(mel_input, verbose=0)[0] for m in cnn_models]
    mel_avg_prob = np.mean(cnn_preds, axis=0)

    mfcc_feat = extract_mfcc_features(signal)
    mfcc_feat = pad_mfcc(mfcc_feat)
    mfcc_input = mfcc_feat[np.newaxis, ..., np.newaxis]

    bilstm_prob    = bilstm_model.predict(mfcc_input, verbose=0)[0]
    emoformer_prob = emoformer_model.predict(mfcc_input, verbose=0)[0]
    mfcc_avg_prob  = (bilstm_prob + emoformer_prob) / 2.0

    final_prob = (mel_avg_prob + mfcc_avg_prob) / 2.0

    predicted_idx   = int(np.argmax(final_prob))
    predicted_label = EMOTIONS[predicted_idx]

    confidence_scores = {
        EMOTIONS[i]: float(round(final_prob[i] * 100, 2))
        for i in range(len(EMOTIONS))
    }

    return {
        "emotion": predicted_label,
        "confidence": confidence_scores,
    }