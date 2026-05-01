import os
import io
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import joblib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pydub import AudioSegment
from pathlib import Path

# ================= PATHS (Dynamic Setup) =================
# BASE_DIR هنا هي الفولدر الرئيسي للبروجيكت (backend)
BASE_DIR = Path(__file__).resolve().parent.parent

# تحديد مسار فولدر الداتا بشكل نسبي
folder_path = BASE_DIR / "Data" / "Arabic_Audio_Deepfake"

# تحميل البيانات (استخدام الـ Path بيضمن إن المسار يشتغل على أي OS)
train_path = folder_path / "train-00000-of-00003.parquet"

# ================= 2. تحميل البيانات =================
if not train_path.exists():
    print(f"❌ Error: File not found at {train_path}")
else:
    df_train = pd.read_parquet(str(train_path))

def extract_mfcc_from_file(audio_path, n_mfcc=13):
    waveform, sr = sf.read(audio_path)
    if len(waveform.shape) > 1:
        waveform = waveform.mean(axis=1)
    if sr != 16000:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
        sr = 16000
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean

# ================= 3. استخراج الميزات (Features) =================
X, y = [], []
for i in tqdm(range(len(df_train))):
    try:
        audio_bytes = df_train.iloc[i]["audio"]["bytes"]
        audio, sr = sf.read(io.BytesIO(audio_bytes))

        target_length = 3 * 16000
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        X.append(np.mean(mfcc, axis=1))
        y.append(df_train.iloc[i]["label"])
    except Exception as e:
        print(f"Error in sample {i}: {e}")

X, y = np.array(X), np.array(y)

# ================= 4. التدريب =================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# ================= 5. حفظ الموديل =================
# الحفظ هيتم جوه فولدر src عشان الـ api.py بيقرأهم من هناك
save_dir = BASE_DIR / "src"
joblib.dump(model, save_dir / "audio_rf_model.pkl")
joblib.dump(scaler, save_dir / "audio_scaler.pkl")

print(f"✅ Model and scaler saved successfully in {save_dir}!")
