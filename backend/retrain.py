"""
retrain.py — Handles saving new user-submitted audio for continuous
learning, and provides a function to fine-tune models periodically.
"""

import os
import shutil
import uuid
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "new_audio")
os.makedirs(DATA_DIR, exist_ok=True)


def save_new_audio(audio_path: str, predicted_emotion: str, user_confirmed_emotion: str = None) -> str:
    """
    Saves a copy of the user's audio file into the continuous-learning
    dataset, organized by the CONFIRMED emotion (if user gave feedback)
    or the model's prediction otherwise.

    This builds up a growing dataset over time that can be used to
    periodically retrain / fine-tune the models.
    """
    # Use user-confirmed label if provided (more reliable), else model's prediction
    label = user_confirmed_emotion if user_confirmed_emotion else predicted_emotion

    emotion_dir = os.path.join(DATA_DIR, label)
    os.makedirs(emotion_dir, exist_ok=True)

    timestamp   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    unique_id   = uuid.uuid4().hex[:8]
    ext         = os.path.splitext(audio_path)[1] or ".wav"
    new_filename = f"{timestamp}_{unique_id}{ext}"
    dest_path    = os.path.join(emotion_dir, new_filename)

    shutil.copy2(audio_path, dest_path)
    return dest_path


def get_new_data_counts() -> dict:
    """Returns how many new audio samples have been collected per emotion."""
    counts = {}
    if not os.path.exists(DATA_DIR):
        return counts
    for emotion in os.listdir(DATA_DIR):
        emo_path = os.path.join(DATA_DIR, emotion)
        if os.path.isdir(emo_path):
            counts[emotion] = len([
                f for f in os.listdir(emo_path)
                if f.lower().endswith((".wav", ".mp3", ".m4a"))
            ])
    return counts


# ── NOTE on full retraining ───────────────────────────────────────────────────
# Full model retraining (re-running the whole notebook pipeline) is too heavy
# to do on every request, or even on a free HuggingFace Space (no GPU, limited
# RAM/CPU). The practical approach for "continuous learning" in production is:
#
#   1. Collect new labeled audio here automatically (this file does this)
#   2. Periodically (e.g. weekly), download data/new_audio/ from the Space
#   3. Merge it into your Colab dataset and re-run training there (GPU)
#   4. Upload the newly trained .h5 models back to replace the old ones
#
# This keeps the live API fast and free, while still "learning" from real
# user data over time. True automatic in-production retraining typically
# requires paid GPU infra, which isn't realistic for a free-tier deployment.
