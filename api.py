from fastapi import FastAPI, Query
import joblib
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

app = FastAPI()

# 📌 Ensure the models directory exists
models_path = "models"
os.makedirs(models_path, exist_ok=True)

# 📌 Model filenames
models = {
    "epc_cost": os.path.join(models_path, "epc_cost_model.pkl"),
    "grid_forecast": os.path.join(models_path, "grid_forecasting.pkl"),
    "oil_gas": os.path.join(models_path, "oil_gas_monitoring.pkl"),
    "procurement": os.path.join(models_path, "procurement_ai.pkl"),
    "maintenance": os.path.join(models_path, "predictive_maintenance.pkl"),
}

# 📌 Check if models exist, otherwise train them
def check_and_train_models():
    for name, path in models.items():
        if not os.path.exists(path):
            print(f"⚠️ Model {name} not found! Training a new one...")
            train_model(name)

# 📌 Training function for all models
def train_model(model_name):
    print(f"🔄 Training {model_name} model...")

    if model_name == "epc_cost":
        data = pd.DataFrame({
            "Project_Size_MW": np.random.randint(50, 500, 500),
            "Material_Cost": np.random.uniform(100, 600, 500),
            "Labor_Cost": np.random.uniform(50, 300, 500),
            "Equipment_Cost": np.random.uniform(30, 200, 500),
            "Total_Cost": np.random.uniform(200, 1000, 500)
        })
        target = "Total_Cost"

    elif model_name == "grid_forecast":
        data = pd.DataFrame({
            "Hour": np.arange(0, 500),
            "Load_MW": np.random.uniform(500, 3000, 500),
            "Temperature": np.random.uniform(15, 45, 500),
            "Humidity": np.random.uniform(20, 90, 500)
        })
        target = "Load_MW"

    elif model_name == "oil_gas":
        data = pd.DataFrame({
            "Pipeline_Age_Years": np.random.randint(1, 50, 500),
            "Pressure_Drop": np.random.uniform(0.1, 10.0, 500),
            "Flow_Rate": np.random.uniform(100, 10000, 500),
            "Leak_Risk": np.random.uniform(0, 1, 500)
        })
        target = "Leak_Risk"

    elif model_name == "procurement":
        data = pd.DataFrame({
            "Material_Type": np.random.choice(["Steel", "Cement", "Cables"], 500),
            "Price_Per_Unit": np.random.uniform(100, 500, 500),
            "Logistics_Cost": np.random.uniform(50, 200, 500),
            "Total_Cost": np.random.uniform(150, 700, 500)
        })
        data = pd.get_dummies(data, columns=["Material_Type"])
        target = "Total_Cost"

    elif model_name == "maintenance":
        data = pd.DataFrame({
            "Machine_Age_Years": np.random.randint(1, 20, 500),
            "Usage_Hours": np.random.randint(100, 10000, 500),
            "Vibration": np.random.uniform(0.1, 5.0, 500),
            "Temperature": np.random.uniform(30, 100, 500),
            "Failure_Probability": np.random.uniform(0, 1, 500)
        })
        target = "Failure_Probability"

    else:
        print(f"❌ Unknown model: {model_name}")
        return

    X = data.drop(columns=[target])
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, models[model_name])
    print(f"✅ {model_name} model trained and saved!")

# 📌 Load all models
check_and_train_models()
loaded_models = {name: joblib.load(path) for name, path in models.items()}

# 📌 ChatGPT-style AI Response API
@app.get("/")
async def home():
    return {"message": "🔥 AI API with Auto-Training, Predictions, and More!"}

@app.get("/predict")
async def predict(
    query: str = Query(..., description="Ask anything about EPC, Grid, Oil & Gas, Procurement, or Maintenance.")
):
    # Handle different types of AI responses
    if "epc cost" in query.lower():
        return {"response": "Enter Project Size (MW) to get EPC Cost Prediction."}
    elif "grid load" in query.lower():
        return {"response": "Enter Hour to get Smart Grid Load Forecasting."}
    elif "oil leak" in query.lower():
        return {"response": "Enter Pipeline Age, Pressure Drop, and Flow Rate for Oil & Gas Monitoring."}
    elif "procurement cost" in query.lower():
        return {"response": "Enter Material Type, Price, and Logistics Cost for Procurement AI."}
    elif "maintenance risk" in query.lower():
        return {"response": "Enter Machine Age, Usage Hours, Vibration, and Temperature for Predictive Maintenance."}
    else:
        return {"response": "I can predict EPC costs, grid load, oil leaks, procurement costs, and maintenance risks. Ask me!"}

# 📌 Individual Prediction Endpoints
@app.get("/predict/epc")
async def predict_epc(project_size: float):
    prediction = loaded_models["epc_cost"].predict([[project_size, 300, 150, 100]])[0]
    return {"EPC Cost Prediction": f"₹{prediction:.2f} Cr"}

@app.get("/predict/grid")
async def predict_grid(hour: int):
    prediction = loaded_models["grid_forecast"].predict([[hour, 30, 50]])[0]
    return {"Grid Load Prediction": f"{prediction:.2f} MW"}

@app.get("/predict/oil_gas")
async def predict_oil_gas(pipeline_age: int, pressure_drop: float, flow_rate: float):
    prediction = loaded_models["oil_gas"].predict([[pipeline_age, pressure_drop, flow_rate]])[0]
    return {"Leak Risk": f"{prediction:.2f} Probability"}

@app.get("/predict/procurement")
async def predict_procurement(price_per_unit: float, logistics_cost: float):
    prediction = loaded_models["procurement"].predict([[price_per_unit, logistics_cost, 1, 0, 0]])[0]
    return {"Procurement Cost": f"₹{prediction:.2f} Cr"}

@app.get("/predict/maintenance")
async def predict_maintenance(machine_age: int, usage_hours: int, vibration: float, temperature: float):
    prediction = loaded_models["maintenance"].predict([[machine_age, usage_hours, vibration, temperature]])[0]
    return {"Failure Probability": f"{prediction:.2f}"}

