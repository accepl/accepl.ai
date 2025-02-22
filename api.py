import os
import joblib
from fastapi import FastAPI
import numpy as np
import pandas as pd

# Define the models directory
models_path = "models"
os.makedirs(models_path, exist_ok=True)

# Model filenames
model_files = {
    "epc_cost_model.pkl": "epc_cost_model",
    "grid_forecasting.pkl": "grid_forecasting",
    "oil_gas_monitoring.pkl": "oil_gas_monitoring",
    "predictive_maintenance.pkl": "predictive_maintenance",
    "procurement_ai.pkl": "procurement_ai"
}

# Load models dynamically or train if missing
for model_name, model_script in model_files.items():
    model_path = os.path.join(models_path, model_name)
    
    if not os.path.exists(model_path):
        print(f"⚠️ {model_name} not found. Training model...")
        exec(f"from {model_script} import train_model; train_model()")  # Train model dynamically
        print(f"✅ {model_name} trained and saved.")

# Load trained models
epc_model = joblib.load(os.path.join(models_path, "epc_cost_model.pkl"))
grid_model = joblib.load(os.path.join(models_path, "grid_forecasting.pkl"))
oil_gas_model = joblib.load(os.path.join(models_path, "oil_gas_monitoring.pkl"))
maintenance_model = joblib.load(os.path.join(models_path, "predictive_maintenance.pkl"))
procurement_model = joblib.load(os.path.join(models_path, "procurement_ai.pkl"))

# Initialize FastAPI
app = FastAPI()

# Endpoint to predict EPC Cost
@app.post("/predict/epc")
def predict_epc(Project_Size_MW: float, Material_Cost: float, Labor_Cost: float, Equipment_Cost: float):
    input_data = np.array([[Project_Size_MW, Material_Cost, Labor_Cost, Equipment_Cost]])
    prediction = epc_model.predict(input_data)
    return {"EPC_Cost_Estimate": prediction[0]}

# Endpoint to predict Grid Load Forecasting
@app.post("/predict/grid")
def predict_grid(Hour: int, Temperature: float, Humidity: float):
    input_data = np.array([[Hour, Temperature, Humidity]])
    prediction = grid_model.predict(input_data)
    return {"Grid_Load_Forecast": prediction[0]}

# Endpoint to predict Oil & Gas Monitoring Risk
@app.post("/predict/oil_gas")
def predict_oil_gas(Pipeline_Age_Years: int, Pressure_Drop: float, Flow_Rate: float):
    input_data = np.array([[Pipeline_Age_Years, Pressure_Drop, Flow_Rate]])
    prediction = oil_gas_model.predict(input_data)
    return {"Leak_Risk_Score": prediction[0]}

# Endpoint to predict Predictive Maintenance Risk
@app.post("/predict/maintenance")
def predict_maintenance(Machine_Age_Years: int, Usage_Hours: int, Vibration: float, Temperature: float):
    input_data = np.array([[Machine_Age_Years, Usage_Hours, Vibration, Temperature]])
    prediction = maintenance_model.predict(input_data)
    return {"Failure_Probability": prediction[0]}

# Endpoint to predict Procurement Cost
@app.post("/predict/procurement")
def predict_procurement(Price_Per_Unit: float, Logistics_Cost: float):
    input_data = np.array([[Price_Per_Unit, Logistics_Cost]])
    prediction = procurement_model.predict(input_data)
    return {"Procurement_Cost_Estimate": prediction[0]}

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to Accepl.AI Prediction API"}
