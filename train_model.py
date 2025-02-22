import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_model(filename, target_column, model_name):
    print(f"🚀 Training AI Model: {model_name}...")
    data = pd.read_csv(f"data/{filename}")
    X = data.drop(columns=[target_column])
    y = data[target_column]

    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    joblib.dump(model, model_path)
    print(f"✅ {model_name} Model Trained & Saved: {model_path}")

# 🚀 Train All Models
def train_all_models():
    train_model("energy_load.csv", "power_demand", "energy_grid")
    train_model("bess_data.csv", "charge_decision", "bess")
    train_model("finance_data.csv", "revenue_forecast", "finance")
    train_model("workforce_data.csv", "optimized_allocation", "workforce")
    train_model("risk_data.csv", "risk_level", "risk")
    print("✅✅✅ ALL AI MODELS TRAINED & READY!")

if __name__ == "__main__":
    train_all_models()
