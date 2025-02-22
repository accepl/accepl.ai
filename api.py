from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
import pandas as pd
import uvicorn
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import requests
from bs4 import BeautifulSoup

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Automatically Train Model if None Exists
def train_model():
    np.random.seed(42)
    X = np.random.rand(1000, 5)  # Synthetic Data
    y = X[:, 0] * 100 + np.random.rand(1000) * 10  # Synthetic Target
    
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    return model

model = train_model()

# Prediction API Endpoint
class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float
    feature5: float

@app.post("/predict")
def predict(data: InputData):
    input_features = np.array([[data.feature1, data.feature2, data.feature3, data.feature4, data.feature5]])
    prediction = model.predict(input_features)[0]
    return {"prediction": prediction}

# Web Search API Endpoint
@app.get("/search")
def search(query: str):
    url = f"https://www.google.com/search?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    results = [link.text for link in soup.find_all("h3")]
    return {"results": results[:5]}  # Return top 5 results

# Web-Based UI
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Accepl.AI</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; }
            img { width: 200px; margin-bottom: 20px; }
            input, button { padding: 10px; margin: 5px; }
        </style>
    </head>
    <body>
        <img src="/static/logo.jpg" alt="Accepl.AI Logo">
        <h1>AI Predictions & Web Search</h1>
        <form action="/predict" method="post">
            <input type="text" name="feature1" placeholder="Feature 1" required>
            <input type="text" name="feature2" placeholder="Feature 2" required>
            <input type="text" name="feature3" placeholder="Feature 3" required>
            <input type="text" name="feature4" placeholder="Feature 4" required>
            <input type="text" name="feature5" placeholder="Feature 5" required>
            <button type="submit">Predict</button>
        </form>
        <form action="/search" method="get">
            <input type="text" name="query" placeholder="Search the web" required>
            <button type="submit">Search</button>
        </form>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
