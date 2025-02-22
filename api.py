from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
import joblib
import requests
from duckduckgo_search import ddg
import textblob
import re

app = FastAPI()

# Load AI Models (Ensure these exist in models/)
models = {
    "epc": joblib.load("models/epc_cost_model.pkl"),
    "grid": joblib.load("models/grid_load_model.pkl"),
    "oil_gas": joblib.load("models/oil_gas_model.pkl"),
    "telecom": joblib.load("models/telecom_model.pkl"),
    "ipp": joblib.load("models/ipp_model.pkl"),
}

# Typo Correction
def correct_query(query):
    blob = textblob.TextBlob(query)
    return str(blob.correct())

# Web Search
def web_search(query):
    results = ddg(query, max_results=3)
    return results if results else "No relevant search results found."

# AI Prediction Handler
def predict(query):
    query_lower = query.lower()

    if "epc" in query_lower:
        return models["epc"].predict([[1]])[0]
    elif "grid" in query_lower:
        return models["grid"].predict([[1]])[0]
    elif "oil" in query_lower or "gas" in query_lower:
        return models["oil_gas"].predict([[1]])[0]
    elif "telecom" in query_lower:
        return models["telecom"].predict([[1]])[0]
    elif "ipp" in query_lower:
        return models["ipp"].predict([[1]])[0]
    
    return "No matching AI prediction found."

# Frontend UI
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Accepl.AI</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
            input { width: 80%; padding: 10px; margin: 10px; }
            button { padding: 10px; cursor: pointer; }
            #response { margin-top: 20px; text-align: left; }
        </style>
    </head>
    <body>
        <h1>Accepl.AI - AI Predictions & Web Search</h1>
        <img src="static/logo.jpg" alt="Logo" width="200">
        
        <input type="text" id="userInput" placeholder="Enter your query...">
        <button onclick="getResponse()">Ask AI</button>

        <div id="response"></div>

        <script>
            async function getResponse() {
                let input = document.getElementById('userInput').value;
                let responseDiv = document.getElementById('response');

                responseDiv.innerHTML = "Thinking...";

                let res = await fetch(`/query?input_text=${encodeURIComponent(input)}`);
                let data = await res.json();

                responseDiv.innerHTML = `
                    <h3>AI Prediction:</h3> <p>${data.ai_prediction}</p>
                    <h3>Web Search:</h3> <p>${data.web_search[0]?.title || "No results"}</p>
                    <p>${data.web_search[0]?.href || ""}</p>
                    <h3>Corrected Query:</h3> <p>${data.query_corrected}</p>
                `;
            }
        </script>
    </body>
    </html>
    """

@app.get("/query")
def ai_query(input_text: str = Query(..., min_length=3)):
    # Correct Typo
    corrected = correct_query(input_text)

    # Search Web
    search_results = web_search(corrected)

    # AI Prediction
    ai_response = predict(corrected)

    return {
        "query_original": input_text,
        "query_corrected": corrected,
        "ai_prediction": ai_response,
        "web_search": search_results
    }
