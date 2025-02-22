import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load AI Models
MODEL_DIR = "models"
models = {
    "energy_grid": joblib.load(os.path.join(MODEL_DIR, "energy_grid.pkl")),
    "bess": joblib.load(os.path.join(MODEL_DIR, "bess_model.pkl")),
    "maintenance": joblib.load(os.path.join(MODEL_DIR, "maintenance_model.pkl")),
    "finance": joblib.load(os.path.join(MODEL_DIR, "finance_model.pkl")),
    "risk": joblib.load(os.path.join(MODEL_DIR, "risk_model.pkl")),
    "workforce": joblib.load(os.path.join(MODEL_DIR, "workforce_model.pkl"))
}

app = FastAPI(title="🔥 Accepl.AI MVP", description="AI for Industrial, Financial, and Energy Decision-Making", version="2.0")

@app.get("/")
def home():
    return {"status": "✅ Accepl.AI MVP is Running!", "message": "AI is ready to automate industrial decision-making."}

@app.get("/predict/{model_name}")
def run_model(model_name: str, feature1: float, feature2: float, feature3: float):
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found!")

    try:
        features = np.array([[feature1, feature2, feature3]])
        prediction = models[model_name].predict(features)[0]
        return {"model": model_name, "prediction": round(prediction, 2)}
    except Exception as e:
        logger.error(f"Error in {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed for {model_name}")

@app.get("/energy-grid-forecast")
def forecast_energy_grid():
    return run_model("energy_grid", 12, 7, 250)

@app.get("/bess-optimization")
def bess_trading():
    return run_model("bess", 3, 25, 12)

@app.get("/predictive-maintenance")
def industrial_maintenance():
    return run_model("maintenance", 0.3, 0.7, 0.2)

@app.get("/financial-forecast")
def financial_analysis():
    return run_model("finance", 2026, 600, 350)

@app.get("/risk-assessment")
def risk_analysis():
    return run_model("risk", 7, 15, 5)

@app.get("/workforce-allocation")
def workforce_optimization():
    return run_model("workforce", 800, 95, 30)
