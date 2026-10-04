from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import List

# ---------------------------------------------------------
# Load Model C + Encoders (NEW)
# ---------------------------------------------------------
model = joblib.load("modelC_treatment_pattern.pkl")
mlb = joblib.load("modelC_methods_mlb.pkl")
le_cluster = joblib.load("modelC_cluster_encoder.pkl")
le_focus = joblib.load("modelC_focus_encoder.pkl")

app = FastAPI(title="Model C Treatment Pattern API")


# ---------------------------------------------------------
# Request schema
# ---------------------------------------------------------
class ModelCRequest(BaseModel):
    disorder: str
    symptoms: List[float]
    mood: float
    sleep: float
    activity: float
    stress: float
    progress: float
    adherence: float
    emotion: str


# ---------------------------------------------------------
# Helper: Build DataFrame row
# ---------------------------------------------------------
def build_input_row(req: ModelCRequest):
    return pd.DataFrame({
        "Symptom Severity (1-10)": [float(np.mean(req.symptoms))],
        "Mood Score (1-10)": [req.mood],
        "Sleep Quality (1-10)": [req.sleep],
        "Physical Activity (hrs/week)": [req.activity],
        "Stress Level (1-10)": [req.stress],
        "Treatment Progress (1-10)": [req.progress],
        "Adherence to Treatment (%)": [req.adherence],
        "Diagnosis": [req.disorder],
        "AI-Detected Emotional State": [req.emotion]
    })


# ---------------------------------------------------------
# Main endpoint (UPDATED)
# ---------------------------------------------------------
@app.post("/predict_model_c")
def predict_model_c(req: ModelCRequest):

    # Build input row
    X = build_input_row(req)

    # Predict
    pred = model.predict(X)[0]

    # Decode cluster + focus
    cluster = le_cluster.inverse_transform([int(pred[0])])[0]
    focus = le_focus.inverse_transform([int(pred[1])])[0]

    # Decode multi-label methods
    method_flags = np.array(pred[2:], dtype=int).reshape(1, -1)
    methods = mlb.inverse_transform(method_flags)[0]

    return {
        "cluster": cluster,
        "focus": focus,
        "methods": list(methods)
    }

print(">>> USING NEW MODEL C SERVER <<<")
