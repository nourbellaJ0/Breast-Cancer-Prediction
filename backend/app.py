# backend/app.py
import os
from typing import Dict, Any, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache
import shap
import numpy as np


# -------- Paths --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../Projet/backend
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))    # .../Projet

SCALER_PATH = os.path.join(PROJECT_DIR, "artifacts", "scaler.pkl")
MODEL_PATHS = {
    "Softmax": os.path.join(PROJECT_DIR, "artifacts", "softmax.pkl"),
    "SVM": os.path.join(PROJECT_DIR, "artifacts", "svm.pkl"),
    "MLP": os.path.join(PROJECT_DIR, "artifacts", "mlp.pkl"),
    "XGBoost": os.path.join(PROJECT_DIR, "artifacts", "xgboost.pkl"),
}
DATA_PATH   = os.path.join(PROJECT_DIR, "artifacts", "data.csv")  # si ton csv est dans artifacts


# -------- API --------
app = FastAPI(title="Prediction API", version="2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en prod: mettre l'URL de ton front
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
scaler = None
models = {}
explainers = {}
FEATURES: List[str] = []


# -------- Utils --------
def to_python(x):
    if hasattr(x, "item"):
        return x.item()
    return x


@lru_cache(maxsize=1)
def load_feature_stats(csv_path: str = DATA_PATH) -> Dict[str, Dict[str, float]]:
    """
    Stats par feature pour randomize réaliste (min, q1, median, q3, max, mean, std).
    Cache en mémoire pour éviter de relire le CSV à chaque requête.
    """
    if not os.path.exists(csv_path):
        return {}

    df = pd.read_csv(csv_path)

    for col in ["id", "diagnosis", "target", "label", "Unnamed: 32"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    desc = df.describe().T
    stats: Dict[str, Dict[str, float]] = {}

    for feat, row in desc.iterrows():
        stats[str(feat)] = {
            "min": float(row["min"]),
            "q1": float(row["25%"]),
            "median": float(row["50%"]),
            "q3": float(row["75%"]),
            "max": float(row["max"]),
            "mean": float(row["mean"]),
            "std": float(row["std"]),
        }
    return stats


# -------- Schemas --------
class PredictRequest(BaseModel):
    features: Dict[str, float] = Field(..., description="features {col: valeur} (raw / non standardisées)")
    model: Optional[str] = Field("Softmax", description="Model to use: Softmax, SVM, MLP, XGBoost")


class PredictResponse(BaseModel):
    prediction: Any
    proba: Optional[float] = None
    proba_by_class: Optional[Dict[str, float]] = None
    used_features: List[str]
    model_used: str
    shap_explanation: Optional[Dict[str, float]] = None


# -------- Startup --------
@app.on_event("startup")
def startup():
    global models, scaler, FEATURES, explainers

    if not os.path.exists(SCALER_PATH):
        raise RuntimeError(f"Scaler introuvable: {SCALER_PATH}")

    scaler = joblib.load(SCALER_PATH)

    for name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
            print(f"Loaded model: {name}")
        else:
            print(f"Warning: Model {name} not found at {path}")

    if not models:
        raise RuntimeError("Aucun modèle chargé.")

    # Use the first model to get features
    first_model = next(iter(models.values()))
    if not hasattr(scaler, "feature_names_in_"):
        raise RuntimeError("Ton scaler ne contient pas feature_names_in_. Réentraîne avec sklearn récent.")

    FEATURES = list(scaler.feature_names_in_)

    # Load background data for SHAP
    if os.path.exists(DATA_PATH):
        background_df = pd.read_csv(DATA_PATH).head(100)  # small sample for background
        for col in ["id", "diagnosis", "target", "label", "Unnamed: 32"]:
            if col in background_df.columns:
                background_df = background_df.drop(columns=[col])
        # Ensure only features known to scaler
        background_df = background_df[FEATURES]
        background_scaled = scaler.transform(background_df)
        for name, model in models.items():
            explainers[name] = shap.KernelExplainer(model.predict_proba, background_scaled)
    else:
        explainers = {}


# -------- Endpoints --------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/feature_names")
def feature_names():
    return {"features": FEATURES, "count": len(FEATURES)}


@app.get("/feature_stats")
def feature_stats():
    """
    Expose les stats du dataset (utile pour Streamlit randomize personnalisé).
    """
    stats = load_feature_stats()
    return {"stats": stats, "count": len(stats)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not models or scaler is None:
        raise HTTPException(status_code=500, detail="Modèles/scaler non chargés.")

    model_name = req.model
    if model_name not in models:
        raise HTTPException(status_code=400, detail=f"Modèle '{model_name}' non disponible. Disponibles: {list(models.keys())}")

    model = models[model_name]

    try:
        missing = [c for c in FEATURES if c not in req.features]
        if missing:
            raise HTTPException(status_code=422, detail=f"Features manquantes: {missing}")

        X = pd.DataFrame([[req.features[c] for c in FEATURES]], columns=FEATURES).astype(float)
        X_scaled = scaler.transform(X)

        y_pred = to_python(model.predict(X_scaled)[0])

        proba = None
        proba_by_class = None

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_scaled)[0]
            classes = getattr(model, "classes_", list(range(len(probs))))
            proba_by_class = {str(classes[i]): float(to_python(probs[i])) for i in range(len(probs))}

            pred_key = str(y_pred)
            proba = float(proba_by_class[pred_key]) if (pred_key in proba_by_class) else float(max(probs))

        shap_explanation = None
        if model_name in explainers:
            shap_values = explainers[model_name].shap_values(X_scaled)
            # For binary classification, shap_values may be list of 1 or 2 arrays
            if isinstance(shap_values, list):
                if len(shap_values) == 2:
                    shap_for_pred = shap_values[y_pred][0]
                else:
                    shap_for_pred = shap_values[0][0]
            else:
                shap_for_pred = shap_values[0]
            shap_for_pred = np.array(shap_for_pred).flatten()
            shap_explanation = {FEATURES[i]: float(shap_for_pred[i]) for i in range(len(FEATURES))}

        return PredictResponse(
            prediction=y_pred,
            proba=proba,
            proba_by_class=proba_by_class,
            used_features=FEATURES,
            model_used=model_name,
            shap_explanation=shap_explanation,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur prédiction: {e}")
