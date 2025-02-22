import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from fastapi.staticfiles import StaticFiles

# 📌 FastAPI App
app = FastAPI(title="🔥 AI-Powered Predictions API")

# 📌 Serve Static Files (For LOGO & Frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 📌 Ensure models directory exists
os.makedirs("models", exist_ok=True)

# 📌 Model Paths
MODEL_PATHS = {
    "epc_cost": "models/epc_cost_model.pkl",
    "grid_forecasting": "models/grid_forecasting.pkl",
    "oil_gas_monitoring": "models/oil_gas_monitoring.pkl",
    "procurement_ai": "models/procurement_ai.pkl",
    "predictive_maintenance": "models/predictive_maintenance.pkl",
    "anomaly_detection": "models/anomaly_detection.pkl"
}

# 📌 Training Data Generation
def generate_training_data(model_name):
    if model_name == "epc_cost":
        return pd.DataFrame({
            "Project_Size_MW": np.random.randint(50, 500, 1000),
            "Material_Cost": np.random.uniform(100, 600, 1000),
            "Labor_Cost": np.random.uniform(50, 300, 1000),
            "Equipment_Cost": np.random.uniform(30, 200, 1000),
            "Total_Cost": np.random.uniform(200, 1000, 1000)
        }), "Total_Cost"

    elif model_name == "grid_forecasting":
        return pd.DataFrame({
            "Hour": np.arange(0, 1000),
            "Load_MW": np.random.uniform(500, 3000, 1000),
            "Temperature": np.random.uniform(15, 45, 1000),
            "Humidity": np.random.uniform(20, 90, 1000)
        }), "Load_MW"

    elif model_name == "oil_gas_monitoring":
        return pd.DataFrame({
            "Pipeline_Age_Years": np.random.randint(1, 50, 1000),
            "Pressure_Drop": np.random.uniform(0.1, 10.0, 1000),
            "Flow_Rate": np.random.uniform(100, 10000, 1000),
            "Leak_Risk": np.random.uniform(0, 1, 1000)
        }), "Leak_Risk"

    elif model_name == "procurement_ai":
        data = pd.DataFrame({
            "Material_Type": np.random.choice(["Steel", "Cement", "Cables"], 1000),
            "Price_Per_Unit": np.random.uniform(100, 500, 1000),
            "Logistics_Cost": np.random.uniform(50, 200, 1000),
            "Total_Cost": np.random.uniform(150, 700, 1000)
        })
        data = pd.get_dummies(data, columns=["Material_Type"])
        return data, "Total_Cost"

    elif model_name == "predictive_maintenance":
        return pd.DataFrame({
            "Machine_Age_Years": np.random.randint(1, 20, 1000),
            "Usage_Hours": np.random.randint(100, 10000, 1000),
            "Vibration": np.random.uniform(0.1, 5.0, 1000),
            "Temperature": np.random.uniform(30, 100, 1000),
            "Failure_Probability": np.random.uniform(0, 1, 1000)
        }), "Failure_Probability"

    return None, None

# 📌 Train & Save Models
def train_model(model_name):
    data, target = generate_training_data(model_name)
    if data is None:
        print(f"❌ Invalid Model Name: {model_name}")
        return

    X = data.drop(columns=[target])
    y = data[target]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATHS[model_name])
    print(f"✅ Model '{model_name}' trained & saved!")

# 📌 Load or Train Models
def load_or_train_models():
    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            print(f"⚠️ Model '{name}' missing! Training now...")
            train_model(name)
        else:
            print(f"✅ Model '{name}' loaded successfully!")

# 📌 Load All Models
load_or_train_models()

# 📌 Input Data Model
class InputData(BaseModel):
    model_name: str
    data: dict

# 📌 Prediction Endpoint
@app.post("/predict/")
def predict(input_data: InputData):
    model_name = input_data.model_name
    data = input_data.data

    if model_name not in MODEL_PATHS:
        raise HTTPException(status_code=400, detail="Invalid model name!")

    model = joblib.load(MODEL_PATHS[model_name])
    df = pd.DataFrame([data])

    try:
        prediction = model.predict(df)[0]
        return {"model": model_name, "input": data, "prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# 📌 GPT-Like Chat UI
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return """
    <html>
        <head>
            <title>🔥 AI Prediction Platform</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; background-color: #f4f4f4; padding: 20px; }
                .container { width: 50%; margin: auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0px 0px 10px gray; }
                img { width: 100px; }
                input, select { width: 80%; padding: 10px; margin: 10px 0; border-radius: 5px; border: 1px solid gray; }
                button { padding: 10px; border: none; background: blue; color: white; border-radius: 5px; cursor: pointer; }
            </style>
        </head>
        <body>
            <div class="container">
                <img src='/static/logo.jpg' alt="Logo">
                <h1>🔥 AI Prediction Platform</h1>
                <label for="model">Select AI Model:</label>
                <select id="model">
                    <option value="epc_cost">Predict EPC Cost</option>
                    <option value="grid_forecasting">Smart Grid Forecast</option>
                    <option value="oil_gas_monitoring">Oil & Gas Monitoring</option>
                    <option value="procurement_ai">Procurement AI</option>
                    <option value="predictive_maintenance">Predictive Maintenance</option>
                </select>
                <input type="text" id="inputData" placeholder="Enter values (JSON format)">
                <button onclick="predict()">Get Prediction</button>
                <h3 id="result"></h3>
            </div>
            <script>
                function predict() {
                    let model = document.getElementById("model").value;
                    let inputData = document.getElementById("inputData").value;
                    fetch("/predict/", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ model_name: model, data: JSON.parse(inputData) })
                    }).then(response => response.json()).then(data => {
                        document.getElementById("result").innerText = "Prediction: " + data.prediction;
                    }).catch(err => console.error(err));
                }
            </script>
        </body>
    </html>
    """
