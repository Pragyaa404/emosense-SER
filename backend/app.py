"""
app.py — FastAPI server for EmoSense.
Exposes endpoints for:
  - POST /predict        → upload audio, get emotion prediction
  - POST /feedback        → confirm/correct a prediction (for continuous learning)
  - GET  /health           → simple health check
  - GET  /stats             → how much new data has been collected
"""

import os
import tempfile

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from predict import predict_emotion, EMOTIONS
from retrain import save_new_audio, get_new_data_counts

app = FastAPI(title="EmoSense API", version="1.0")

# ── CORS — allow your Vercel frontend to call this API ───────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # in production, replace * with your exact Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok", "emotions": EMOTIONS}


@app.get("/stats")
def stats():
    return {"new_audio_collected": get_new_data_counts()}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an audio file upload, runs the 5-model ensemble,
    saves the audio for continuous learning, and returns the prediction.
    """
    # Save uploaded file to a temp path so librosa can read it
    suffix = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = predict_emotion(tmp_path)

        # Save this audio into the continuous-learning dataset,
        # labeled with the model's own prediction for now
        # (gets corrected later if user submits feedback)
        save_new_audio(tmp_path, predicted_emotion=result["emotion"])

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        os.remove(tmp_path)


@app.post("/feedback")
async def feedback(
    file: UploadFile = File(...),
    predicted_emotion: str = Form(...),
    correct_emotion: str = Form(...),
):
    """
    Lets the user correct a wrong prediction. This re-saves the audio
    under the CORRECT label, which is much more valuable for future
    retraining than an unconfirmed model guess.
    """
    suffix = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        if correct_emotion not in EMOTIONS:
            return JSONResponse(
                status_code=400,
                content={"error": f"correct_emotion must be one of {EMOTIONS}"}
            )

        save_new_audio(
            tmp_path,
            predicted_emotion=predicted_emotion,
            user_confirmed_emotion=correct_emotion
        )
        return {"status": "feedback saved", "label": correct_emotion}

    finally:
        os.remove(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
