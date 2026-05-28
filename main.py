from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional  
import tensorflow as tf
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
# 1. DEFINISI CUSTOM LAYER (DIAMBIL DARI FILE TRAINING ANDA)
# =========================================================
class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, units, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.units        = units
        self.dropout_rate = dropout_rate
        self.dense1     = tf.keras.layers.Dense(units, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.bn1        = tf.keras.layers.BatchNormalization()
        self.dropout1   = tf.keras.layers.Dropout(dropout_rate)
        self.dense2     = tf.keras.layers.Dense(units, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.bn2        = tf.keras.layers.BatchNormalization()
        self.projection = tf.keras.layers.Dense(units)

    def call(self, inputs, training=False):
        x        = self.dense1(inputs)
        x        = self.bn1(x, training=training)
        x        = self.dropout1(x, training=training)
        x        = self.dense2(x)
        x        = self.bn2(x, training=training)
        shortcut = self.projection(inputs)
        return tf.keras.activations.relu(x + shortcut)

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units, 'dropout_rate': self.dropout_rate})
        return config


# 2. KONFIGURASI JALUR FILE (PATH) SECARA DINAMIS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_multioutput_final.keras")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "preprocessor_risklevel.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder_risklevel.pkl")

loaded_model = None
preprocessor = None
label_encoder = None

# 3. PROSES MEMUAT ASET (DENGAN REGISTER CUSTOM OBJECT)
try:
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print("✓ Preprocessor_risklevel.pkl berhasil dimuat.")
except Exception as e:
    print(f"[ERROR CRITICAL] Gagal memuat Preprocessor! Detail: {str(e)}")
    raise RuntimeError(f"Preprocessor crash: {str(e)}")

try:
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    print("✓ Label_encoder_risklevel.pkl berhasil dimuat.")
except Exception as e:
    print(f"[ERROR CRITICAL] Gagal memuat Label Encoder! Detail: {str(e)}")
    raise RuntimeError(f"Label Encoder crash: {str(e)}")

try:
    # Registrasi ResidualBlock dimasukkan ke sini agar tidak eror deserialisasi
    loaded_model = tf.keras.models.load_model(
        MODEL_PATH, 
        compile=False,
        custom_objects={'ResidualBlock': ResidualBlock}
    )
    print("✓ Model_multioutput_final.keras berhasil dimuat dengan Custom Layer.")
except Exception as e:
    print(f"[ERROR CRITICAL] Gagal memuat Model .keras! Detail: {str(e)}")
    raise RuntimeError(f"Model crash: {str(e)}")


# 4. SKEMA DATA INPUT
class EmployeeInput(BaseModel):
    Age: int
    Gender: str
    JobRole: str
    Experience: int
    WorkHoursPerWeek: float
    RemoteRatio: float
    SatisfactionLevel: float
    StressLevel: int


@app.get("/")
def home():
    return {"message": "API Burnout Multi-Output aktif dan seluruh aset termuat sempurna!"}


@app.post("/predict")
def predict_burnout(data: EmployeeInput):
    try:
        if preprocessor is None or loaded_model is None or label_encoder is None:
            raise HTTPException(status_code=500, detail="Aset model belum termuat sempurna di server.")

        # Ekstrak data input dan ubah menjadi DataFrame
        input_dict = data.model_dump()
        new_employee_data = pd.DataFrame([input_dict])
        
        # Transformasi data
        new_employee_processed = preprocessor.transform(new_employee_data)
        
        # Hitung prediksi model AI
        predictions = loaded_model.predict(new_employee_processed, verbose=0)
        
        # Membaca output berdasarkan format penamaan di model Anda (Dictionary / Named Output)
        if isinstance(predictions, dict):
            burnout_output = predictions['burnout_output']
            risk_output = predictions['risk_output']
        else:
            # Antisipasi jika tensorflow mengembalikannya dalam bentuk list sesuai urutan output-layer
            burnout_output = predictions[0]
            risk_output = predictions[1]
        
        # 1. Ambil nilai probabilitas burnout
        prediction_prob = float(burnout_output[0][0])
        burnout_prob_percent_numeric = prediction_prob * 100  
        
        # 2. Ambil tingkat risiko (Risk Level) menggunakan Label Encoder
        risk_class_idx = int(np.argmax(risk_output[0]))
        status = str(label_encoder.inverse_transform([risk_class_idx])[0])
        
        # Mapping Tema Warna & Saran Otomatis berdasarkan teks dari Label Encoder
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
        warna_hex = meta_status["warna"]

        # Logika Rekomendasi Kesehatan Kerja (Wellness)
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