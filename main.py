from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import keras
import pandas as pd
import joblib
import numpy as np
import os

app = FastAPI(title="Burnout Detection API (Multi-Output Fixed)", version="2.0")

# =========================================================
# KONFIGURASI CORS
# =========================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# CUSTOM LAYER
# =========================================================
@keras.saving.register_keras_serializable(package="custom_layers")
class ResidualBlock(keras.layers.Layer):
    def __init__(self, units, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.units        = units
        self.dropout_rate = dropout_rate
        self.dense1     = keras.layers.Dense(units, activation='relu',
                              kernel_regularizer=keras.regularizers.l2(1e-4))
        self.bn1        = keras.layers.BatchNormalization()
        self.dropout1   = keras.layers.Dropout(dropout_rate)
        self.dense2     = keras.layers.Dense(units, activation='relu',
                              kernel_regularizer=keras.regularizers.l2(1e-4))
        self.bn2        = keras.layers.BatchNormalization()
        self.projection = keras.layers.Dense(units)

    def call(self, inputs, training=False):
        x        = self.dense1(inputs)
        x        = self.bn1(x, training=training)
        x        = self.dropout1(x, training=training)
        x        = self.dense2(x)
        x        = self.bn2(x, training=training)
        shortcut = self.projection(inputs)
        return keras.activations.relu(x + shortcut)

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units, 'dropout_rate': self.dropout_rate})
        return config

# =========================================================
# PATH FILE
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH       = os.path.join(BASE_DIR, "model_multioutput_final.keras")
PREPROCESSOR_PATH  = os.path.join(BASE_DIR, "preprocessor_risklevel.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder_risklevel.pkl")

loaded_model  = None
preprocessor  = None
label_encoder = None

# =========================================================
# LOAD ASET
# =========================================================
try:
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print("✓ Preprocessor berhasil dimuat.")
except Exception as e:
    raise RuntimeError(f"Preprocessor crash: {str(e)}")

try:
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    print("✓ Label Encoder berhasil dimuat.")
except Exception as e:
    raise RuntimeError(f"Label Encoder crash: {str(e)}")

try:
    loaded_model = keras.models.load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={'ResidualBlock': ResidualBlock}
    )
    print("✓ Model berhasil dimuat.")
except Exception as e:
    raise RuntimeError(f"Model crash: {str(e)}")


# =========================================================
# SKEMA INPUT
# =========================================================
class EmployeeInput(BaseModel):
    Age: int
    Gender: str
    JobRole: str
    Experience: int
    WorkHoursPerWeek: float
    RemoteRatio: float
    SatisfactionLevel: float
    StressLevel: int


# =========================================================
# ENDPOINTS
# =========================================================
@app.get("/")
def home():
    return {"message": "API Burnout Multi-Output aktif dan seluruh aset termuat sempurna!"}


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": loaded_model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "label_encoder_loaded": label_encoder is not None,
    }


@app.post("/predict")
def predict_burnout(data: EmployeeInput):
    try:
        if preprocessor is None or loaded_model is None or label_encoder is None:
            raise HTTPException(status_code=500, detail="Aset model belum termuat sempurna di server.")

        input_dict = data.model_dump()
        new_employee_data = pd.DataFrame([input_dict])

        new_employee_processed = preprocessor.transform(new_employee_data)

        predictions = loaded_model.predict(new_employee_processed, verbose=0)

        if isinstance(predictions, dict):
            burnout_output = predictions['burnout_output']
            risk_output    = predictions['risk_output']
        else:
            burnout_output = predictions[0]
            risk_output    = predictions[1]

        prediction_prob              = float(burnout_output[0][0])
        burnout_prob_percent_numeric = prediction_prob * 100

        risk_class_idx = int(np.argmax(risk_output[0]))
        status         = str(label_encoder.inverse_transform([risk_class_idx])[0])

        status_mapping = {
            "Risiko Rendah": {
                "saran": "Karyawan dalam kondisi sehat dan seimbang. Tetap pertahankan lingkungan kerja yang positif.",
                "warna": "#28A745"
            },
            "Risiko Sedang": {
                "saran": "Kondisi kerja perlu dimonitor secara berkala agar tidak berkembang menjadi burnout serius.",
                "warna": "#F39C12"
            },
            "Risiko Tinggi": {
                "saran": "Segera lakukan tindakan intervensi, berikan waktu istirahat tambahan, atau jadwalkan sesi counseling.",
                "warna": "#E74C3C"
            }
        }

        meta_status = status_mapping.get(status, {
            "saran": "Evaluasi kondisi kerja karyawan secara berkala untuk mencegah stres berlebih.",
            "warna": "#F39C12"
        })
        saran_umum = meta_status["saran"]
        warna_hex  = meta_status["warna"]

        ai_recommendation = []
        if data.StressLevel >= 8:
            ai_recommendation.append("Lakukan mindfulness atau breathing exercise 10–15 menit setiap hari.")
            ai_recommendation.append("Kurangi multitasking berlebihan agar fokus kerja lebih stabil.")
        elif data.StressLevel >= 6:
            ai_recommendation.append("Cobalah mengatur ulang prioritas kerja agar tekanan kerja lebih terkontrol.")

        if data.WorkHoursPerWeek >= 55:
            ai_recommendation.append("Kurangi lembur dan prioritaskan work-life balance.")
            ai_recommendation.append("Hindari bekerja tanpa jeda terlalu lama.")
        elif data.WorkHoursPerWeek >= 50:
            ai_recommendation.append("Pantau jam kerja mingguan agar tetap dalam batas sehat.")

        if data.SatisfactionLevel <= 2.0:
            ai_recommendation.append("Diskusikan hambatan kerja dengan HR atau atasan.")
            ai_recommendation.append("Cari aktivitas kerja yang meningkatkan engagement.")
        elif data.SatisfactionLevel <= 3.0:
            ai_recommendation.append("Evaluasi faktor yang menyebabkan kepuasan kerja menurun.")

        if data.RemoteRatio <= 20.0:
            ai_recommendation.append("Hybrid working dapat membantu meningkatkan fleksibilitas kerja.")

        if burnout_prob_percent_numeric >= 75.0:
            ai_recommendation.append("Disarankan mengikuti counseling session profesional.")
            ai_recommendation.append("Kurangi tekanan kerja sementara untuk pemulihan mental.")

        default_tips = [
            "Pastikan tidur cukup minimal 7 jam setiap hari.",
            "Lakukan aktivitas fisik ringan secara rutin.",
            "Jaga pola makan dan hidrasi selama bekerja.",
        ]

        while len(ai_recommendation) < 3:
            ai_recommendation.append(default_tips[len(ai_recommendation) % len(default_tips)])

        final_recommendations = ai_recommendation[:5]

        return {
            "status_code": 200,
            "success": True,
            "data": {
                "burnout_probability": prediction_prob,
                "burnout_probability_percent": f"{prediction_prob:.1%}",
                "risk_level": status,
                "hr_recommendation": saran_umum,
                "ui_theme_color": warna_hex,
                "ai_wellness_recommendations": final_recommendations
            }
        }

    except HTTPException as http_e:
        raise http_e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat inference: {str(e)}")


# =========================================================
# ENTRY POINT (untuk Railway)
# =========================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)