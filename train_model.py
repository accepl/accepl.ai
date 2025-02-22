import joblib
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# ✅ Ensure 'models' directory exists
os.makedirs("models", exist_ok=True)

# ✅ Define function to train and save the best model
def train_and_save_model(name, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize the features for better performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train multiple models
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=200, random_state=42)
    }

    best_model = None
    best_score = -float("inf")

    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        score = r2_score(y_test, model.predict(X_test_scaled))

        if score > best_score:
            best_score = score
            best_model = model

    # Save the best model
    joblib.dump(best_model, f"models/{name}.pkl")
    print(f"✅ Trained {name} model ({type(best_model).__name__}) with R² score: {best_score:.4f}")

# 📌 Train EPC Cost Estimation Model
def train_epc_model():
    print("🔹 Training EPC Cost Model...")
    data = pd.DataFrame({
        "Project_Size_MW": np.random.randint(50, 500, 1000),
        "Material_Cost": np.random.uniform(100, 600, 1000),
        "Labor_Cost": np.random.uniform(50, 300, 1000),
        "Equipment_Cost": np.random.uniform(30, 200, 1000),
        "Total_Cost": lambda df: 0.8 * df["Project_Size_MW"] + 1.2 * df["Material_Cost"] + 1.5 * df["Labor_Cost"] + 1.1 * df["Equipment_Cost"] + np.random.uniform(-50, 50, 1000)
    })
    X = data.drop(columns=["Total_Cost"])
    y = data["Total_Cost"]
    train_and_save_model("epc_cost_model", X, y)

# 📌 Train Smart Grid Load Forecasting Model
def train_smart_grid_model():
    print("🔹 Training Smart Grid Load Forecasting Model...")
    data = pd.DataFrame({
        "Hour": np.random.randint(0, 24, 1000),
        "Load_MW": lambda df: 50 * df["Hour"] + np.random.uniform(500, 3000, 1000),
        "Temperature": np.random.uniform(15, 45, 1000),
        "Humidity": np.random.uniform(20, 90, 1000)
    })
    X = data.drop(columns=["Load_MW"])
    y = data["Load_MW"]
    train_and_save_model("grid_forecasting", X, y)

# 📌 Train Procurement Optimization Model
def train_procurement_model():
    print("🔹 Training Procurement Optimization Model...")
    data = pd.DataFrame({
        "Material_Type": np.random.choice(["Steel", "Cement", "Cables"], 1000),
        "Price_Per_Unit": np.random.uniform(100, 500, 1000),
        "Logistics_Cost": np.random.uniform(50, 200, 1000),
        "Total_Cost": lambda df: df["Price_Per_Unit"] * 1.1 + df["Logistics_Cost"] * 0.8 + np.random.uniform(-20, 20, 1000)
    })
    data = pd.get_dummies(data, columns=["Material_Type"])  # Convert categorical to numerical
    X = data.drop(columns=["Total_Cost"])
    y = data["Total_Cost"]
    train_and_save_model("procurement_ai", X, y)

# 📌 Train Predictive Maintenance Model
def train_maintenance_model():
    print("🔹 Training Predictive Maintenance Model...")
    data = pd.DataFrame({
        "Machine_Age_Years": np.random.randint(1, 30, 1000),
        "Usage_Hours": np.random.randint(100, 20000, 1000),
        "Vibration": np.random.uniform(0.1, 5.0, 1000),
        "Temperature": np.random.uniform(30, 150, 1000),
        "Failure_Probability": lambda df: -0.01 * df["Usage_Hours"] + 0.05 * df["Vibration"] + 0.04 * df["Temperature"] + np.random.uniform(0, 0.2, 1000)
    })
    X = data.drop(columns=["Failure_Probability"])
    y = data["Failure_Probability"]
    train_and_save_model("predictive_maintenance", X, y)

# 📌 Train Oil & Gas Monitoring Model
def train_oil_gas_model():
    print("🔹 Training Oil & Gas Monitoring Model...")
    data = pd.DataFrame({
        "Pipeline_Age_Years": np.random.randint(1, 50, 1000),
        "Pressure_Drop": np.random.uniform(0.1, 10.0, 1000),
        "Flow_Rate": np.random.uniform(100, 20000, 1000),
        "Leak_Risk": lambda df: -0.5 * df["Pressure_Drop"] + 0.3 * df["Flow_Rate"] + 0.1 * df["Pipeline_Age_Years"] + np.random.uniform(-0.2, 0.2, 1000)
    })
    X = data.drop(columns=["Leak_Risk"])
    y = data["Leak_Risk"]
    train_and_save_model("oil_gas_monitoring", X, y)

# 📌 Train All Models
if __name__ == "__main__":
    train_epc_model()
    train_smart_grid_model()
    train_procurement_model()
    train_maintenance_model()
    train_oil_gas_model()
    print("🔥 All AI Models Trained and Saved Successfully! 🚀")
