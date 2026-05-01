import io
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np
import librosa
import joblib
from pathlib import Path
import tensorflow as tf
from PIL import Image


# ================= RAG IMPORTS =================
from rag import (
    load_vector_db,
    load_embedding_model,
    load_generation_model,
    retrieve_candidates,
    build_legal_context,
    generate_answer
)

# ================= APP =================
app = FastAPI(title="Egyptian Law – AI Hub")

# ================= CORS =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= PATHS (Dynamic Setup) =================
# BASE_DIR هنا هي فولدر الـ backend الرئيسي
BASE_DIR = Path(__file__).resolve().parent.parent

# مسارات ملفات الصوت (موجودة جوه src)
MODEL_PATH = BASE_DIR / "src" / "audio_rf_model.pkl"
SCALER_PATH = BASE_DIR / "src" / "audio_scaler.pkl"

# مسار موديل الصور (Data/Face_Models)
IMAGE_MODEL_PATH = BASE_DIR / "Data" / "Face_Models" / "best_pretrained_model2.h5"

# ================= LOAD RAG MODELS =================
print("⏳ Loading Legal Brain...")
index, metadata = load_vector_db()
emb_model = load_embedding_model()
gen_model = load_generation_model("gemma3:4b")
print("✅ Legal RAG Ready!")

# ================= LOAD AUDIO MODELS =================
print("⏳ Loading Audio Models...")
audio_model = joblib.load(MODEL_PATH)
audio_scaler = joblib.load(SCALER_PATH)

# ================= LOAD IMAGE MODELS =================
print("⏳ Loading Image Model...")
image_model = tf.keras.models.load_model(IMAGE_MODEL_PATH)

# ================= RAG API =================
class LegalQuery(BaseModel):
    question: str

@app.post("/ask")
async def handle_query(query: LegalQuery):
    candidates = retrieve_candidates(
        query.question, index, metadata, emb_model
    )
    context = build_legal_context(candidates)
    answer = generate_answer(
        gen_model, context, query.question, candidates
    )

    return {
        "answer": answer,
        "sources": candidates
    }

# ================= AUDIO DEEPFAKE PART =================
def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=16000)
    target_length = 3 * 16000
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]

    mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=40)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    combined = np.hstack([
        np.mean(mfcc, axis=1),
        np.mean(delta_mfcc, axis=1),
        np.mean(delta2_mfcc, axis=1)
    ])
    return combined.reshape(1, -1)

@app.post("/detect-audio")
async def detect_audio(file: UploadFile = File(...)):
    features = extract_features(file.file)
    features_scaled = audio_scaler.transform(features)

    prediction = audio_model.predict(features_scaled)[0]
    proba = audio_model.predict_proba(features_scaled)[0]

    return {
        "result": "Fake / AI" if prediction == 1 else "Real Human",
        "confidence": round(float(max(proba)) * 100, 2),
        "status": "Verified" if max(proba) > 0.85 and prediction == 0 else "Suspicious"
    }

# ================= IMAGE DEEPFAKE PART =================
def preprocess_image(image_file):
    img = Image.open(image_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/detect-image")
async def detect_image(file: UploadFile = File(...)):
    image = preprocess_image(file.file)
    prediction = image_model.predict(image)[0][0]

    if prediction >= 0.5:
        result = "Real"
        confidence = prediction
    else:
        result = "Fake"
        confidence = 1 - prediction

    return {
        "result": result,
        "confidence": round(float(confidence) * 100, 2),
        "disclaimer": "AI-assisted forensic analysis – not a final legal judgment"
    }

# ================= RUN =================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860) # تعديل بورت Hugging Face الافتراضي
