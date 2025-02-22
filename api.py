import os
import joblib
import numpy as np
import logging
import requests
from fastapi import FastAPI, HTTPException
from web import search  # ✅ Web Search API for market data

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(filename=os.path.join(LOG_DIR, "acceplai.log"), level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = "models"
models = {
    "epc": joblib.load(os.path.join(MODEL_DIR, "epc_model.pkl")),
    "smart_grid": joblib.load(os.path.join(MODEL_DIR, "smart_grid_model.pkl")),
    "telecom": joblib.load(os.path.join(MODEL_DIR, "telecom_model.pkl")),
    "oil_gas": joblib.load(os.path.join(MODEL_DIR, "oil_gas_model.pkl")),
    "finance": joblib.load(os.path.join(MODEL_DIR, "financial_model.pkl")),
}

app = FastAPI(title="🔥 Accepl.AI MVP", description="AI for EPC, Energy, Telecom, and Financial Automation", version="3.0")

@app.get("/")
def home():
    return {"status": "✅ Accepl.AI MVP is Running!"}

@app.get("/epc-cost/{project_type}/{capacity_mw}")
def epc_cost(project_type: str, capacity_mw: float):
    cost_per_mw = {"solar": 4, "wind": 6, "thermal": 8, "hydro": 10}.get(project_type.lower(), None)
    if cost_per_mw is None:
        raise HTTPException(status_code=400, detail="Invalid project type. Use 'solar', 'wind', 'thermal', 'hydro'.")
    
    total_cost = cost_per_mw * capacity_mw
    avg_tariff = float(search(f"Current {project_type} energy tariff in India"))  # Web search for real tariff data
    roi = round((avg_tariff * 25 * capacity_mw) / total_cost, 2)  # Approx ROI for 25-year lifecycle
    irr = round((roi * 5) / 2, 2)  # Rough IRR estimation

    return {
        "project_type": project_type,
        "capacity_mw": capacity_mw,
        "total_cost_in_crores": total_cost,
        "avg_market_tariff": avg_tariff,
        "roi_25_years": roi,
        "irr": irr
    }
