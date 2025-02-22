from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
import joblib
import os
import pandas as pd
import numpy as np
from pydantic import BaseModel
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Load models (Auto-load on startup)
models_path = "models"
epc_model = joblib.load(os.path.join(models_path, "epc_cost_model.pkl"))
grid_model = joblib.load(os.path.join(models_path, "grid_forecasting.pkl"))
predictive_maintenance_model = joblib.load(os.path.join(models_path, "predictive_maintenance.pkl"))
oil_gas_model = joblib.load(os.path.join(models_path, "oil_gas_monitoring.pkl"))
procurement_model = joblib.load(os.path.join(models_path, "procurement_ai.pkl"))

# Serve UI with Logo
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Accepl.AI - Industrial AI Chat</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; margin: 40px; }
        img { width: 200px; }
        input, button { font-size: 18px; padding: 10px; }
        .chat-box { width: 60%; margin: auto; }
        .response { font-weight: bold; margin-top: 20px; }
    </style>
</head>
<body>
    <img src="/static/logo.jpg" alt="Accepl.AI Logo">
    <h1>Industrial AI Chat</h1>
    <form action="/ask" method="post">
        <input type="text" name="query" placeholder="Ask AI..." required>
        <button type="submit">Submit</button>
    </form>
    <div class="response">{{response}}</div>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return html_content.replace("{{response}}", "")

# AI Chat Logic
@app.post("/ask", response_class=HTMLResponse)
def ask_ai(query: str = Form(...)):
    response = handle_query(query)
    return html_content.replace("{{response}}", response)

# AI Model Decision Logic
def handle_query(query):
    query = query.lower()
    
    if "epc cost" in query:
        project_size = np.random.randint(50, 500)  # Simulate input
        prediction = epc_model.predict([[project_size]])[0]
        return f"Predicted EPC Cost for {project_size} MW: ₹{round(prediction, 2)} crore"
    
    elif "grid load" in query:
        hour = np.random.randint(0, 24)  # Simulate input
        prediction = grid_model.predict([[hour]])[0]
        return f"Predicted Grid Load at Hour {hour}: {round(prediction, 2)} MW"
    
    elif "maintenance" in query:
        failure_score = predictive_maintenance_model.predict([[np.random.random()]])[0]
        return f"Predictive Maintenance Risk Score: {round(failure_score, 2)} (Higher means riskier)"
    
    elif "oil gas" in query:
        sensor_data = np.random.random(size=(1, 5))
        prediction = oil_gas_model.predict(sensor_data)[0]
        return f"Oil & Gas Monitoring AI Output: {prediction}"
    
    elif "procurement" in query:
        demand_forecast = procurement_model.predict([[np.random.randint(10, 1000)]])[0]
        return f"AI Procurement Recommendation: Order {round(demand_forecast, 2)} units"
    
    else:
        return "AI Response: Sorry, I am trained for EPC, Grid, Maintenance, Oil & Gas, and Procurement insights."

# Run API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
