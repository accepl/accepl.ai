from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Template
import joblib
import os
import numpy as np
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import ddg
from textblob import TextBlob
import re

# Initialize FastAPI app
app = FastAPI()

# Serve static files (for logo)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load trained models (Replace with actual trained models)
models = {
    "epc": joblib.load("epc_cost_model.pkl") if os.path.exists("epc_cost_model.pkl") else None,
    "grid": joblib.load("grid_load_model.pkl") if os.path.exists("grid_load_model.pkl") else None,
    "oil_gas": joblib.load("oil_gas_model.pkl") if os.path.exists("oil_gas_model.pkl") else None,
    "telecom": joblib.load("telecom_model.pkl") if os.path.exists("telecom_model.pkl") else None,
    "ipp": joblib.load("ipp_model.pkl") if os.path.exists("ipp_model.pkl") else None,
}

# Jinja2 Template for Frontend (Chat UI)
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Accepl.AI - AI Chat & Predictions</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            margin: 0;
            padding: 20px;
            background-color: #f4f4f4;
        }
        .container {
            max-width: 600px;
            margin: auto;
            padding: 20px;
            background: white;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        }
        img {
            width: 150px;
            margin-bottom: 10px;
        }
        input {
            width: 80%;
            padding: 10px;
            margin-top: 10px;
            border-radius: 5px;
            border: 1px solid #ddd;
        }
        button {
            padding: 10px 20px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            margin-top: 10px;
        }
        .response {
            margin-top: 20px;
            padding: 10px;
            background: #e9ecef;
            border-radius: 5px;
            text-align: left;
        }
    </style>
</head>
<body>
    <div class="container">
        <img src="/static/logo.jpg" alt="Accepl.AI Logo">
        <h2>Accepl.AI - AI Chat & Predictions</h2>
        <form action="/" method="post">
            <input type="text" name="query" placeholder="Ask AI anything..." required>
            <button type="submit">Ask</button>
        </form>
        {% if response %}
        <div class="response">
            <h3>AI Response:</h3>
            <p>{{ response }}</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

# Function to handle predictions
def predict(query):
    query_lower = query.lower()

    # Handle EPC cost predictions
    if "epc cost" in query_lower:
        match = re.search(r'\d+', query)
        mw = int(match.group()) if match else 50  # Default to 50MW
        if models["epc"]:
            prediction = models["epc"].predict([[mw]])
            return f"Estimated EPC cost for {mw} MW: ₹{round(prediction[0], 2)} crore."
        return "EPC cost model not available."

    # Handle Grid Load Forecast
    elif "grid load" in query_lower:
        match = re.search(r'\d+', query)
        hour = int(match.group()) if match else 12  # Default to 12 PM
        if models["grid"]:
            prediction = models["grid"].predict([[hour]])
            return f"Predicted grid load at hour {hour}: {round(prediction[0], 2)} MW."
        return "Grid load model not available."

    # Handle Oil & Gas Forecasting
    elif "oil price" in query_lower or "gas price" in query_lower:
        if models["oil_gas"]:
            prediction = models["oil_gas"].predict([[1]])
            return f"Projected Oil & Gas price: ${round(prediction[0], 2)} per barrel."
        return "Oil & Gas model not available."

    # Handle Telecom Predictions
    elif "5g" in query_lower or "subscriber" in query_lower:
        if models["telecom"]:
            prediction = models["telecom"].predict([[1]])
            return f"Projected Telecom ARPU: ₹{round(prediction[0], 2)} per user."
        return "Telecom model not available."

    # Web Search for Real-time Data
    elif "search" in query_lower or "news" in query_lower:
        results = ddg(query, max_results=3)
        return "\n".join([f"{r['title']}: {r['href']}" for r in results])

    # Handle typos using TextBlob
    else:
        corrected = TextBlob(query).correct()
        if str(corrected).lower() != query_lower:
            return f"Did you mean: {corrected}?"
        return "I'm not sure. Try asking about EPC costs, grid load, oil prices, telecom, or search the web."

# Route for rendering the chat frontend
@app.get("/", response_class=HTMLResponse)
async def get_homepage():
    return Template(html_template).render(response=None)

# Route for processing user input
@app.post("/", response_class=HTMLResponse)
async def ask_ai(request: Request, query: str = Form(...)):
    response = predict(query)
    return Template(html_template).render(response=response)
