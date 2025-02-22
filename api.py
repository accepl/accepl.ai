from fastapi import FastAPI, Query
import joblib
import os
import requests
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Initialize FastAPI app
app = FastAPI()

# Load trained models (EPC, Grid Load, Oil & Gas, Telecom, IPP)
models_path = "models"
epc_model = joblib.load(os.path.join(models_path, "epc_cost_model.pkl"))
grid_model = joblib.load(os.path.join(models_path, "grid_forecasting.pkl"))
oil_gas_model = joblib.load(os.path.join(models_path, "oil_gas_monitoring.pkl"))
telecom_model = joblib.load(os.path.join(models_path, "telecom_cost_model.pkl"))
ipp_model = joblib.load(os.path.join(models_path, "ipp_financial_model.pkl"))

# Web Search Function (for real-time data fetching)
def web_search(query):
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key=YOUR_API_KEY"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()["items"][0]["snippet"]
    return "No relevant data found."

# Error Handling Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AI-based Financial Predictions
def predict_financials(model, features):
    try:
        prediction = model.predict([features])
        return float(prediction[0])
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return "Error processing prediction."

@app.get("/")
def read_root():
    return {"message": "🔥 Accepl.AI Advanced AI Chat & Predictions"}

# EPC Cost Prediction
@app.get("/predict/epc_cost")
def predict_epc_cost(project_size: float = Query(..., description="Project size in MW")):
    predicted_cost = predict_financials(epc_model, [project_size])
    return {"EPC Cost Prediction (₹ Crore)": predicted_cost}

# Grid Load Forecasting
@app.get("/predict/grid_load")
def predict_grid_load(hour: int = Query(..., description="Hour of the day (0-23)")):
    predicted_load = predict_financials(grid_model, [hour])
    return {"Predicted Grid Load (MW)": predicted_load}

# Oil & Gas Financial Projections
@app.get("/predict/oil_gas")
def predict_oil_gas(reserve_estimate: float = Query(..., description="Reserve estimate in million barrels")):
    projected_cost = predict_financials(oil_gas_model, [reserve_estimate])
    return {"Projected Oil & Gas Cost ($M)": projected_cost}

# Telecom Cost Analysis
@app.get("/predict/telecom")
def predict_telecom(subscribers: int = Query(..., description="Number of subscribers (millions)")):
    telecom_revenue = predict_financials(telecom_model, [subscribers])
    return {"Projected Telecom Revenue ($B)": telecom_revenue}

# IPP Financials
@app.get("/predict/ipp")
def predict_ipp(ppa_tariff: float = Query(..., description="PPA tariff (₹/kWh)")):
    ipp_cost = predict_financials(ipp_model, [ppa_tariff])
    return {"IPP Projected Cost (₹ Crore)": ipp_cost}

# Web Search for Real-Time Data
@app.get("/search")
def search_web(query: str = Query(..., description="Search any industry-related query")):
    return {"Web Search Result": web_search(query)}

# AI Chat (Dynamic Responses)
@app.get("/chat")
def chat_ai(prompt: str = Query(..., description="Ask any question about EPC, IPP, Oil & Gas, or Telecom")):
    try:
        if "epc" in prompt.lower():
            return {"AI Response": "EPC (Engineering, Procurement, Construction) involves managing costs, contracts, and project execution."}
        elif "grid" in prompt.lower():
            return {"AI Response": "Grid load prediction helps balance electricity demand and supply."}
        elif "oil" in prompt.lower():
            return {"AI Response": "Oil & Gas projects require reserve estimates, drilling costs, and market forecasts."}
        elif "telecom" in prompt.lower():
            return {"AI Response": "Telecom investments involve spectrum licensing, network rollout, and subscriber retention strategies."}
        else:
            return {"AI Response": web_search(prompt)}
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return {"AI Response": "Error processing request."}

# Error Handling
@app.exception_handler(Exception)
def handle_exceptions(request, exc):
    logger.error(f"Unhandled Error: {exc}")
    return {"error": "Internal Server Error. Please try again."}
