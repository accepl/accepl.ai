import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ✅ Initialize FastAPI App
app = FastAPI()

# ✅ Serve Static Files (HTML, CSS, Logo, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ✅ Serve Logo Correctly
@app.get("/logo")
async def get_logo():
    """ Returns the company logo """
    return FileResponse("static/logo.jpg")

# ✅ Define Model Paths
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)  # Ensure models directory exists

MODEL_FILES = {
    "epc": "epc_cost_model.pkl",
    "grid": "grid_forecasting.pkl",
    "oil_gas": "oil_gas_monitoring.pkl",
    "procurement": "procurement_ai.pkl",
    "maintenance": "predictive_maintenance.pkl"
}

# ✅ Function to Train & Save Models
def train_models():
    print("🔥 Training AI Models...")

    # 🏗️ Train EPC Cost Model
    print("🛠️ Training EPC Cost Model...")
    epc_data = pd.DataFrame({
        "Project_Size_MW": np.random.randint(50, 500, 500),
        "Material_Cost": np.random.uniform(100, 600, 500),
        "Labor_Cost": np.random.uniform(50, 300, 500),
        "Equipment_Cost": np.random.uniform(30, 200, 500),
        "Total_Cost": np.random.uniform(200, 1000, 500)
    })
    X = epc_data.drop(columns=["Total_Cost"])
    y = epc_data["Total_Cost"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    epc_model = RandomForestRegressor(n_estimators=100, random_state=42)
    epc_model.fit(X_train, y_train)
    joblib.dump(epc_model, os.path.join(MODELS_DIR, MODEL_FILES["epc"]))

    # 🔌 Train Smart Grid Load Forecasting Model
    print("⚡ Training Smart Grid Load Forecasting Model...")
    grid_data = pd.DataFrame({
        "Hour": np.arange(0, 500),
        "Load_MW": np.random.uniform(500, 3000, 500),
        "Temperature": np.random.uniform(15, 45, 500),
        "Humidity": np.random.uniform(20, 90, 500)
    })
    X = grid_data.drop(columns=["Load_MW"])
    y = grid_data["Load_MW"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    grid_model = RandomForestRegressor(n_estimators=100, random_state=42)
    grid_model.fit(X_train, y_train)
    joblib.dump(grid_model, os.path.join(MODELS_DIR, MODEL_FILES["grid"]))

    # 🛢️ Train Oil & Gas Monitoring Model
    print("⛽ Training Oil & Gas Monitoring Model...")
    oil_gas_data = pd.DataFrame({
        "Pipeline_Age_Years": np.random.randint(1, 50, 500),
        "Pressure_Drop": np.random.uniform(0.1, 10.0, 500),
        "Flow_Rate": np.random.uniform(100, 10000, 500),
        "Leak_Risk": np.random.uniform(0, 1, 500)
    })
    X = oil_gas_data.drop(columns=["Leak_Risk"])
    y = oil_gas_data["Leak_Risk"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    oil_gas_model = RandomForestRegressor(n_estimators=100, random_state=42)
    oil_gas_model.fit(X_train, y_train)
    joblib.dump(oil_gas_model, os.path.join(MODELS_DIR, MODEL_FILES["oil_gas"]))

    # 📦 Train Procurement Optimization Model
    print("📊 Training Procurement Optimization Model...")
    procurement_data = pd.DataFrame({
        "Material_Type": np.random.choice(["Steel", "Cement", "Cables"], 500),
        "Price_Per_Unit": np.random.uniform(100, 500, 500),
        "Logistics_Cost": np.random.uniform(50, 200, 500),
        "Total_Cost": np.random.uniform(150, 700, 500)
    })
    procurement_data = pd.get_dummies(procurement_data, columns=["Material_Type"])
    X = procurement_data.drop(columns=["Total_Cost"])
    y = procurement_data["Total_Cost"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    procurement_model = RandomForestRegressor(n_estimators=100, random_state=42)
    procurement_model.fit(X_train, y_train)
    joblib.dump(procurement_model, os.path.join(MODELS_DIR, MODEL_FILES["procurement"]))

    # ⚙️ Train Predictive Maintenance Model
    print("🛠️ Training Predictive Maintenance Model...")
    maintenance_data = pd.DataFrame({
        "Machine_Age_Years": np.random.randint(1, 20, 500),
        "Usage_Hours": np.random.randint(100, 10000, 500),
        "Vibration": np.random.uniform(0.1, 5.0, 500),
        "Temperature": np.random.uniform(30, 100, 500),
        "Failure_Probability": np.random.uniform(0, 1, 500)
    })
    X = maintenance_data.drop(columns=["Failure_Probability"])
    y = maintenance_data["Failure_Probability"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    maintenance_model = RandomForestRegressor(n_estimators=100, random_state=42)
    maintenance_model.fit(X_train, y_train)
    joblib.dump(maintenance_model, os.path.join(MODELS_DIR, MODEL_FILES["maintenance"]))

    print("✅ All Models Trained & Saved Successfully!")

# ✅ Train Models Before Running API
train_models()

# ✅ Load Trained Models
models = {}
for model_name, model_file in MODEL_FILES.items():
    model_path = os.path.join(MODELS_DIR, model_file)
    models[model_name] = joblib.load(model_path)

# ✅ Define Input Format for Predictions
class PredictionInput(BaseModel):
    data: dict  # Expecting input as a dictionary

# ✅ Serve Homepage
@app.get("/", response_class=HTMLResponse)
async def serve_home():
    """ Serve HTML Frontend """
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)

# ✅ Prediction API
@app.post("/predict/{model_name}")
async def predict(model_name: str, input_data: PredictionInput):
    """ Run prediction on selected AI model """
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model `{model_name}` not found!")

    try:
        model = models[model_name]
        df = pd.DataFrame([input_data.data])  # Convert input to DataFrame
        prediction = model.predict(df)
        return JSONResponse(content={"prediction": prediction.tolist()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
