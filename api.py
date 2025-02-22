import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split

# 📌 FastAPI App
app = FastAPI(title="🔥 AI-Powered EPC, Smart Grid & Industrial Predictions API")

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

    elif model_name == "anomaly_detection":
        return pd.DataFrame({
            "Sensor_1": np.random.normal(0, 1, 1000),
            "Sensor_2": np.random.normal(0, 1, 1000),
            "Sensor_3": np.random.normal(0, 1, 1000),
            "Anomaly_Score": np.random.uniform(0, 1, 1000)
        }), "Anomaly_Score"

    return None, None

# 📌 Train & Save Models
def train_model(model_name):
    data, target = generate_training_data(model_name)
    if data is None:
        print(f"❌ Invalid Model Name: {model_name}")
        return

    X = data.drop(columns=[target])
    y = data[target]

    if model_name == "anomaly_detection":
        model = IsolationForest(contamination=0.05)
    else:
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

# 📌 Retrain Model
@app.post("/retrain/")
def retrain(model_name: str):
    if model_name not in MODEL_PATHS:
        raise HTTPException(status_code=400, detail="Invalid model name!")

    train_model(model_name)
    return {"status": "success", "message": f"Model '{model_name}' retrained successfully!"}

# 📌 Root Endpoint
@app.get("/")
def home():
    return {"message": "🔥 AI API with Auto-Training, Predictions, and More!"}
