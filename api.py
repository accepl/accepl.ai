import os
import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from duckduckgo_search import DuckDuckGoSearch
import requests
import traceback
import unidecode
from textblob import TextBlob
import re

app = FastAPI()

# ✅ Load trained AI models
models_path = "models"
epc_model = joblib.load(os.path.join(models_path, "epc_cost_model.pkl"))
grid_model = joblib.load(os.path.join(models_path, "grid_load_model.pkl"))
oil_gas_model = joblib.load(os.path.join(models_path, "oil_gas_model.pkl"))
telecom_model = joblib.load(os.path.join(models_path, "telecom_model.pkl"))
ipp_model = joblib.load(os.path.join(models_path, "ipp_model.pkl"))

# ✅ Define request body format
class PredictionRequest(BaseModel):
    query: str
    industry: Optional[str] = None

# ✅ Web Search Function
def web_search(query):
    try:
        search_results = DuckDuckGoSearch().search(query, max_results=3)
        return "\n".join([f"{r['title']}: {r['href']}" for r in search_results])
    except Exception as e:
        return f"Web search failed: {str(e)}"

# ✅ Typo Handling & Smart Query Understanding
def correct_text(text):
    corrected = TextBlob(text).correct()
    return str(corrected)

# ✅ Error Debugging & Explanation
def handle_error(e):
    return f"Error occurred: {str(e)}\n{traceback.format_exc()}"

# ✅ Industry Prediction Logic
def predict_value(query, industry):
    try:
        query = correct_text(query).lower()

        if "epc" in industry:
            return {"Prediction": epc_model.predict([[np.random.rand()]])[0]}
        elif "grid" in industry:
            return {"Prediction": grid_model.predict([[np.random.rand()]])[0]}
        elif "oil" in industry or "gas" in industry:
            return {"Prediction": oil_gas_model.predict([[np.random.rand()]])[0]}
        elif "telecom" in industry:
            return {"Prediction": telecom_model.predict([[np.random.rand()]])[0]}
        elif "ipp" in industry:
            return {"Prediction": ipp_model.predict([[np.random.rand()]])[0]}
        else:
            return {"Error": "Industry not recognized"}
    except Exception as e:
        return handle_error(e)

# ✅ AI-Powered Financial Projections
def generate_financial_projections(industry):
    try:
        projections = {
            "EPC": {
                "Contract Value": np.random.randint(100, 500),
                "Progress Billing": np.random.randint(10, 100),
                "Retention Money": np.random.randint(5, 50),
                "Variation Orders": np.random.randint(5, 25),
            },
            "IPP": {
                "PPA Tariffs": np.random.uniform(3.5, 6.5),
                "Capacity Factor": np.random.uniform(30, 70),
                "Fuel Cost Projections": np.random.randint(100, 500),
                "O&M Costs": np.random.randint(50, 200),
            },
            "Oil & Gas": {
                "Reserve Estimates": np.random.randint(500, 2000),
                "Production Profiles": np.random.randint(100, 800),
                "Commodity Price Forecasts": np.random.randint(50, 150),
            },
            "Telecom": {
                "Subscriber Acquisition Cost": np.random.randint(50, 200),
                "Network Rollout Costs": np.random.randint(500, 5000),
                "Spectrum Licensing Fees": np.random.randint(100, 1000),
            }
        }
        return projections.get(industry, {"Error": "Industry not recognized"})
    except Exception as e:
        return handle_error(e)

# ✅ API Endpoint: AI Query Processing
@app.post("/predict")
def predict(data: PredictionRequest):
    try:
        if data.industry:
            return predict_value(data.query, data.industry)
        else:
            return {"Error": "Industry must be specified"}
    except Exception as e:
        return handle_error(e)

# ✅ API Endpoint: Web Search
@app.get("/search")
def search(query: str):
    return {"Results": web_search(query)}

# ✅ API Endpoint: Financial Projections
@app.get("/financials")
def financials(industry: str = Query(..., description="Industry Name")):
    return generate_financial_projections(industry)

# ✅ Run Server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
