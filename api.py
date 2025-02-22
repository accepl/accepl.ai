from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
import os
import joblib
import numpy as np
import pandas as pd
import uvicorn
from sklearn.linear_model import LinearRegression
from jinja2 import Template

# Ensure the models directory exists
models_path = "models"
os.makedirs(models_path, exist_ok=True)

# Initialize FastAPI app
app = FastAPI()

# Load or Train Models
def train_epc_model():
    """Train and save an EPC cost model."""
    model_file = os.path.join(models_path, "epc_cost_model.pkl")
    if not os.path.exists(model_file):
        print("Training EPC model...")
        X = np.array([[10], [20], [50], [100], [200]])  # Example project sizes in MW
        y = np.array([100, 200, 500, 1000, 2000])  # Example EPC costs in million INR
        model = LinearRegression()
        model.fit(X, y)
        joblib.dump(model, model_file)
    return joblib.load(model_file)

def train_grid_forecast_model():
    """Train and save a Smart Grid forecasting model."""
    model_file = os.path.join(models_path, "grid_forecast_model.pkl")
    if not os.path.exists(model_file):
        print("Training Grid Forecast model...")
        X = np.array([[0], [6], [12], [18], [24]])  # Example hours
        y = np.array([500, 700, 1500, 1000, 600])  # Example energy demand
        model = LinearRegression()
        model.fit(X, y)
        joblib.dump(model, model_file)
    return joblib.load(model_file)

# Load models (train if missing)
epc_model = train_epc_model()
grid_model = train_grid_forecast_model()

# Load logo
logo_path = "/static/logo.jpg"

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Accepl.AI - AI Predictions</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; margin: 50px; background-color: #f4f4f4; }
        h1 { color: #333; }
        img { width: 150px; margin-bottom: 20px; }
        input, button { padding: 10px; margin: 5px; }
        .chat-box { width: 50%; margin: auto; background: white; padding: 20px; border-radius: 10px; }
    </style>
</head>
<body>
    <img src="{{ logo_path }}" alt="Logo">
    <h1>Accepl.AI - AI Chat & Predictions</h1>
    
    <div class="chat-box">
        <h3>Ask AI:</h3>
        <form method="post" action="/">
            <input type="text" name="prompt" placeholder="Type your question..." required>
            <button type="submit">Ask</button>
        </form>
        <p><strong>AI Response:</strong> {{ response }}</p>
    </div>
</body>
</html>
"""

# Simple AI Response System
def process_prompt(prompt: str):
    prompt = prompt.lower()

    if "epc cost" in prompt:
        try:
            value = int("".join(filter(str.isdigit, prompt)))
            prediction = epc_model.predict(np.array([[value]]))[0]
            return f"Estimated EPC cost for {value} MW is ₹{prediction:.2f} million."
        except:
            return "Please specify a valid project size in MW."

    elif "grid load" in prompt:
        try:
            value = int("".join(filter(str.isdigit, prompt)))
            prediction = grid_model.predict(np.array([[value]]))[0]
            return f"Predicted energy demand at hour {value}: {prediction:.2f} MW."
        except:
            return "Please specify a valid hour (0-24)."

    elif "hello" in prompt:
        return "Hey there! Ask me about EPC costs, grid forecasts, or AI predictions!"

    return "I'm not sure. Try asking about EPC costs or grid load predictions."

# Route to serve the main UI
@app.get("/", response_class=HTMLResponse)
async def main():
    template = Template(HTML_TEMPLATE)
    return template.render(response="", logo_path=logo_path)

# Handle user input and AI responses
@app.post("/", response_class=HTMLResponse)
async def chat(request: Request, prompt: str = Form(...)):
    response = process_prompt(prompt)
    template = Template(HTML_TEMPLATE)
    return template.render(response=response, logo_path=logo_path)

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
