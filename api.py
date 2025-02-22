import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles

# ✅ Define models directory
models_path = os.path.join(os.path.dirname(__file__), "models")

# ✅ Load trained models
epc_model = joblib.load(os.path.join(models_path, "epc_cost_model.pkl"))
grid_model = joblib.load(os.path.join(models_path, "grid_forecasting.pkl"))
procurement_model = joblib.load(os.path.join(models_path, "procurement_ai.pkl"))
maintenance_model = joblib.load(os.path.join(models_path, "predictive_maintenance.pkl"))
oil_gas_model = joblib.load(os.path.join(models_path, "oil_gas_monitoring.pkl"))

# ✅ Initialize FastAPI app
app = FastAPI()

# ✅ Serve static files (for frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 📌 Define API input structure
class EPCInput(BaseModel):
    project_size_mw: float
    material_cost: float
    labor_cost: float
    equipment_cost: float

class GridInput(BaseModel):
    hour: int
    temperature: float
    humidity: float

class ProcurementInput(BaseModel):
    price_per_unit: float
    logistics_cost: float
    material_type_steel: int
    material_type_cement: int
    material_type_cables: int

class MaintenanceInput(BaseModel):
    machine_age_years: int
    usage_hours: int
    vibration: float
    temperature: float

class OilGasInput(BaseModel):
    pipeline_age_years: int
    pressure_drop: float
    flow_rate: float

# 📌 API Endpoints

@app.get("/")
def read_root():
    return {"message": "AI EPC API is running. Visit /static/index.html for UI."}

@app.post("/predict/epc/")
def predict_epc(input_data: EPCInput):
    df = pd.DataFrame([input_data.dict()])
    prediction = epc_model.predict(df)[0]
    return {"estimated_total_cost": round(prediction, 2)}

@app.post("/predict/grid/")
def predict_grid(input_data: GridInput):
    df = pd.DataFrame([input_data.dict()])
    prediction = grid_model.predict(df)[0]
    return {"predicted_load_mw": round(prediction, 2)}

@app.post("/predict/procurement/")
def predict_procurement(input_data: ProcurementInput):
    df = pd.DataFrame([input_data.dict()])
    prediction = procurement_model.predict(df)[0]
    return {"estimated_procurement_cost": round(prediction, 2)}

@app.post("/predict/maintenance/")
def predict_maintenance(input_data: MaintenanceInput):
    df = pd.DataFrame([input_data.dict()])
    prediction = maintenance_model.predict(df)[0]
    return {"failure_probability": round(prediction, 4)}

@app.post("/predict/oil_gas/")
def predict_oil_gas(input_data: OilGasInput):
    df = pd.DataFrame([input_data.dict()])
    prediction = oil_gas_model.predict(df)[0]
    return {"leak_risk_probability": round(prediction, 4)}

print("✅ API is ready and all models are loaded.")
