import logging
import joblib
import requests
from fastapi import FastAPI
from web import search

# ✅ Setup Logging
logging.basicConfig(filename="logs/acceplai_api.log", level=logging.INFO)

# ✅ Load Models
MODEL_DIR = "models"
models = {
    "epc": joblib.load(f"{MODEL_DIR}/epc_model.pkl"),
    "smart_grid": joblib.load(f"{MODEL_DIR}/smart_grid_model.pkl"),
    "telecom": joblib.load(f"{MODEL_DIR}/telecom_model.pkl"),
    "oil_gas": joblib.load(f"{MODEL_DIR}/oil_gas_model.pkl"),
    "finance": joblib.load(f"{MODEL_DIR}/financial_model.pkl"),
}

app = FastAPI(title="🔥 Accepl.AI API", description="AI for EPC, Energy, Telecom, Oil & Gas, and Finance")

@app.get("/")
def home():
    return {"status": "✅ Accepl.AI API Running!"}

@app.get("/ai-query")
def ai_query(query: str):
    if "solar project cost" in query.lower():
        return {"response": "To calculate Solar EPC cost, use /epc-cost/solar/{capacity_mw}"}
    elif "market tariff" in query.lower():
        tariff = search("current electricity tariff in India")
        return {"response": f"Current electricity tariff: {tariff} INR/kWh"}
    else:
        return {"response": "AI Query Not Recognized"}
