import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ✅ Initialize FastAPI App
app = FastAPI()

# ✅ Serve Frontend (HTML, CSS, JS from `static/`)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ✅ Define Model Paths
MODELS_DIR = "models"
MODEL_FILES = {
    "epc": "epc_cost_model.pkl",
    "grid": "grid_forecasting.pkl",
    "oil_gas": "oil_gas_monitoring.pkl",
    "procurement": "procurement_ai.pkl",
    "maintenance": "predictive_maintenance.pkl"
}

# ✅ Load Models with Error Handling
models = {}
for model_name, model_file in MODEL_FILES.items():
    model_path = os.path.join(MODELS_DIR, model_file)
    if os.path.exists(model_path):
        models[model_name] = joblib.load(model_path)
    else:
        print(f"❌ ERROR: Missing model `{model_file}` in `/models/` folder!")

# ✅ Input Data Schemas
class EPCInput(BaseModel):
    project_size_mw: float
    material_cost: float
    labor_cost: float
    equipment_cost: float

class GridInput(BaseModel):
    hour: int
    temperature: float
    humidity: float

class OilGasInput(BaseModel):
    pipeline_age_years: int
    pressure_drop: float
    flow_rate: float

class ProcurementInput(BaseModel):
    price_per_unit: float
    logistics_cost: float
    material_type: str

class MaintenanceInput(BaseModel):
    machine_age_years: int
    usage_hours: int
    vibration: float
    temperature: float

# ✅ API Routes
@app.get("/", response_class=HTMLResponse)
async def serve_home():
    """ Serve HTML Frontend """
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/predict/epc")
async def predict_epc(data: EPCInput):
    """ Predict EPC Cost """
    try:
        model = models.get("epc")
        if not model:
            raise HTTPException(status_code=500, detail="EPC Model Not Found!")
        
        X = np.array([[data.project_size_mw, data.material_cost, data.labor_cost, data.equipment_cost]])
        prediction = model.predict(X)[0]
        return {"epc_cost_prediction": round(prediction, 2)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/predict/grid")
async def predict_grid(data: GridInput):
    """ Predict Grid Load Forecasting """
    try:
        model = models.get("grid")
        if not model:
            raise HTTPException(status_code=500, detail="Grid Model Not Found!")
        
        X = np.array([[data.hour, data.temperature, data.humidity]])
        prediction = model.predict(X)[0]
        return {"grid_load_prediction": round(prediction, 2)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/predict/oil_gas")
async def predict_oil_gas(data: OilGasInput):
    """ Predict Oil & Gas Leak Risk """
    try:
        model = models.get("oil_gas")
        if not model:
            raise HTTPException(status_code=500, detail="Oil & Gas Model Not Found!")
        
        X = np.array([[data.pipeline_age_years, data.pressure_drop, data.flow_rate]])
        prediction = model.predict(X)[0]
        return {"oil_gas_leak_risk": round(prediction, 4)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/predict/procurement")
async def predict_procurement(data: ProcurementInput):
    """ Predict Procurement Cost """
    try:
        model = models.get("procurement")
        if not model:
            raise HTTPException(status_code=500, detail="Procurement Model Not Found!")

        material_encoding = {"Steel": [1, 0, 0], "Cement": [0, 1, 0], "Cables": [0, 0, 1]}
        material_one_hot = material_encoding.get(data.material_type, [0, 0, 0])

        X = np.array([[data.price_per_unit, data.logistics_cost] + material_one_hot])
        prediction = model.predict(X)[0]
        return {"procurement_cost_prediction": round(prediction, 2)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/predict/maintenance")
async def predict_maintenance(data: MaintenanceInput):
    """ Predict Machine Failure Probability """
    try:
        model = models.get("maintenance")
        if not model:
            raise HTTPException(status_code=500, detail="Predictive Maintenance Model Not Found!")

        X = np.array([[data.machine_age_years, data.usage_hours, data.vibration, data.temperature]])
        prediction = model.predict(X)[0]
        return {"failure_probability": round(prediction, 4)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
