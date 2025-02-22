from fastapi import FastAPI, Form
from pydantic import BaseModel
import uvicorn
import requests
import json
import numpy as np
import pandas as pd
import joblib
import traceback
from fastapi.responses import HTMLResponse
from bs4 import BeautifulSoup
from langdetect import detect

# Initialize FastAPI app
app = FastAPI()

# Load trained models (dummy placeholders for now)
models = {
    "epc": joblib.load("models/epc_model.pkl") if "models/epc_model.pkl" else None,
    "grid": joblib.load("models/grid_model.pkl") if "models/grid_model.pkl" else None,
    "oil_gas": joblib.load("models/oil_gas_model.pkl") if "models/oil_gas_model.pkl" else None,
    "telecom": joblib.load("models/telecom_model.pkl") if "models/telecom_model.pkl" else None,
    "ipp": joblib.load("models/ipp_model.pkl") if "models/ipp_model.pkl" else None,
}

# Web-based UI with input prompt
HTML_FORM = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Chat Interface</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 50px; text-align: center; }
        input { width: 400px; padding: 10px; font-size: 16px; }
        button { padding: 10px 20px; font-size: 16px; cursor: pointer; }
        .response { margin-top: 20px; font-weight: bold; }
    </style>
</head>
<body>
    <h1>AI Chat Interface</h1>
    <form action="/query" method="post">
        <input type="text" name="query" placeholder="Ask something..." required>
        <button type="submit">Submit</button>
    </form>
    <div class="response" id="response"></div>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_FORM

@app.post("/query")
async def handle_query(query: str = Form(...)):
    try:
        response = process_query(query)
        return {"response": response}
    except Exception as e:
        return {"error": f"Error processing request: {str(e)}"}

def process_query(query):
    """Processes the user query and routes it accordingly."""
    query_lower = query.lower()

    if "epc" in query_lower:
        return run_prediction("epc", query)
    elif "grid" in query_lower:
        return run_prediction("grid", query)
    elif "oil" in query_lower or "gas" in query_lower:
        return run_prediction("oil_gas", query)
    elif "telecom" in query_lower:
        return run_prediction("telecom", query)
    elif "ipp" in query_lower:
        return run_prediction("ipp", query)
    elif "financial" in query_lower:
        return get_financial_projections()
    elif "search" in query_lower:
        return web_search(query.replace("search", "").strip())
    else:
        return "I couldn't understand your request. Try asking about EPC, Grid, Oil & Gas, Telecom, IPP, or Financial Projections."

def run_prediction(model_name, query):
    """Runs AI prediction for a specific model."""
    try:
        model = models.get(model_name)
        if model:
            sample_input = np.random.rand(1, 10)  # Dummy input, replace with real data
            prediction = model.predict(sample_input)
            return f"{model_name.upper()} Prediction: {prediction.tolist()}"
        else:
            return f"Model for {model_name.upper()} is not available."
    except Exception as e:
        return f"Error running prediction: {str(e)}"

def get_financial_projections():
    """Returns financial projections (Dummy Data)."""
    data = {
        "EPC Revenue": "₹1,50,000 Crore per year",
        "IPP Expansion": "40 GW+",
        "Oil & Gas Revenue": "₹75,000 Crore per year",
        "Telecom Expansion": "₹35,000 Crore per year"
    }
    return json.dumps(data, indent=2)

def web_search(query):
    """Fetches real-time data from the web."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        search_url = f"https://www.google.com/search?q={query}"
        response = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        results = [a.text for a in soup.find_all("h3")[:5]]
        return results if results else "No search results found."
    except Exception as e:
        return f"Web search failed: {str(e)}"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
