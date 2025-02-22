import os
import joblib
import uvicorn
import pandas as pd
import numpy as np
from fastapi import FastAPI, Form, Query
from fastapi.responses import HTMLResponse
from duckduckgo_search import DDGS
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from textblob import TextBlob
from unidecode import unidecode
from typing import Optional

# FastAPI App Initialization
app = FastAPI()

# Ensure models are trained and available
model_path = "epc_cost_model.pkl"

def train_epc_model():
    """Train an EPC Cost Prediction Model and Save it."""
    np.random.seed(42)
    X = np.random.rand(500, 1) * 100  # Project Size in MW
    y = X.flatten() * 1.2 + np.random.rand(500) * 20  # Simulated EPC Costs
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    joblib.dump(model, model_path)
    return model

# Load or Train Model
if os.path.exists(model_path):
    epc_model = joblib.load(model_path)
else:
    epc_model = train_epc_model()

# Web Search

def search_web(query: str):
    """Perform a DuckDuckGo search and return the top result."""
    results = list(DDGS().text(query, max_results=1))
    return results[0]['body'] if results else "No relevant information found."

# Smart Query Processing

def clean_text(text: str):
    """Normalize and correct user input."""
    return unidecode(TextBlob(text).correct().lower().strip())

# HTML UI
html_ui = """
<!DOCTYPE html>
<html>
<head>
    <title>Accepl.AI - AI Chat & Predictions</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; }
        .container { width: 50%; margin: auto; padding: 20px; }
        img { width: 200px; }
        input, button { padding: 10px; margin: 10px; }
    </style>
</head>
<body>
    <img src="/static/logo.jpg" alt="Logo">
    <h1>Accepl.AI - AI Chat & Predictions</h1>
    <form action="/ask" method="get">
        <input type="text" name="question" placeholder="Type your question..." required>
        <button type="submit">Ask</button>
    </form>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def serve_home():
    return html_ui

class QueryRequest(BaseModel):
    question: str

@app.get("/ask")
def ask_ai(question: str = Query(..., min_length=3)):
    """AI Chat Response System."""
    question = clean_text(question)
    
    if "epc cost" in question:
        return {"response": "Enter project size in MW: /predict_epc?size=XX"}
    
    if "search" in question:
        search_term = question.replace("search", "").strip()
        return {"response": search_web(search_term)}
    
    return {"response": "I'm not sure. Try asking about EPC costs or searching the web."}

@app.get("/predict_epc")
def predict_epc(size: float = Query(..., description="Project Size in MW")):
    """Predict EPC cost based on project size."""
    cost = epc_model.predict(np.array([[size]]))[0]
    return {"Project Size (MW)": size, "Predicted EPC Cost (in crores)": round(cost, 2)}

# Run the API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
