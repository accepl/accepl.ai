from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Initialize FastAPI
app = FastAPI()

# Define paths
logo_path = "static/logo.jpg"

# 📌 Train EPC Cost Model (Every time API starts)
def train_epc_model():
    data = pd.DataFrame({
        "Project_Size_MW": np.random.randint(50, 500, 500),
        "Material_Cost": np.random.uniform(100, 600, 500),
        "Labor_Cost": np.random.uniform(50, 300, 500),
        "Equipment_Cost": np.random.uniform(30, 200, 500),
        "Total_Cost": np.random.uniform(200, 1000, 500)
    })
    X = data.drop(columns=["Total_Cost"])
    y = data["Total_Cost"]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# 📌 Train Grid Load Forecasting Model
def train_smart_grid_model():
    data = pd.DataFrame({
        "Hour": np.arange(0, 500),
        "Load_MW": np.random.uniform(500, 3000, 500),
        "Temperature": np.random.uniform(15, 45, 500),
        "Humidity": np.random.uniform(20, 90, 500)
    })
    X = data.drop(columns=["Load_MW"])
    y = data["Load_MW"]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Train models on startup (Directly stored in RAM, no files)
epc_model = train_epc_model()
grid_model = train_smart_grid_model()

# 📌 Homepage UI
@app.get("/", response_class=HTMLResponse)
async def home():
    return f"""
    <html>
    <head>
        <title>AI Models API</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                text-align: center;
                margin: 40px;
            }}
            h1 {{ color: #2c3e50; }}
            img {{ width: 200px; }}
            form {{ margin-top: 20px; }}
        </style>
    </head>
    <body>
        <img src="/logo" alt="Logo">
        <h1>AI Models API</h1>
        <p>Predict EPC Costs and Smart Grid Load Forecasting</p>

        <h2>Predict EPC Cost</h2>
        <form action="/predict_epc" method="get">
            <label>Project Size (MW):</label>
            <input type="number" name="size" required>
            <button type="submit">Predict</button>
        </form>

        <h2>Predict Grid Load</h2>
        <form action="/predict_grid" method="get">
            <label>Hour:</label>
            <input type="number" name="hour" required>
            <button type="submit">Predict</button>
        </form>
    </body>
    </html>
    """

# 📌 Serve Logo
@app.get("/logo")
async def get_logo():
    return FileResponse(logo_path)

# 📌 API Endpoint: Predict EPC Cost
@app.get("/predict_epc")
async def predict_epc(size: int):
    prediction = epc_model.predict([[size, 300, 200, 100]])[0]
    return {"project_size_mw": size, "predicted_total_cost": round(prediction, 2)}

# 📌 API Endpoint: Predict Smart Grid Load
@app.get("/predict_grid")
async def predict_grid(hour: int):
    prediction = grid_model.predict([[hour, 30, 60]])[0]
    return {"hour": hour, "predicted_load_mw": round(prediction, 2)}

# Run locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
