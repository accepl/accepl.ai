import os
import joblib
import pandas as pd
import logging
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# ✅ Logging Setup
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(filename=os.path.join(LOG_DIR, "training.log"), level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_model(filename, target_column, model_name):
    try:
        logger.info(f"🚀 Training Model: {model_name}")
        data = pd.read_csv(f"data/{filename}")
        X = data.drop(columns=[target_column])
        y = data[target_column]

        model = XGBRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
        joblib.dump(model, model_path)
        logger.info(f"✅ Model Trained & Saved: {model_name} at {model_path}")
    except Exception as e:
        logger.error(f"🚨 Training Failed for {model_name}: {str(e)}")

def train_all_models():
    train_model("epc_data.csv", "project_cost", "epc")
    train_model("smart_grid_data.csv", "grid_load", "smart_grid")
    train_model("telecom_data.csv", "network_performance", "telecom")
    train_model("oil_gas_data.csv", "pipeline_failures", "oil_gas")
    train_model("financial_data.csv", "investment_returns", "finance")
    logger.info("✅✅✅ ALL AI MODELS TRAINED & READY!")

if __name__ == "__main__":
    train_all_models()
