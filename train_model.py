import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# 📌 Train EPC Cost Estimation Model
def train_epc_model():
    print("Training EPC Cost Model...")
    data = pd.DataFrame({
        "Project_Size_MW": np.random.randint(50, 500, 500),
        "Material_Cost": np.random.uniform(100, 600, 500),
        "Labor_Cost": np.random.uniform(50, 300, 500),
        "Equipment_Cost": np.random.uniform(30, 200, 500),
        "Total_Cost": np.random.uniform(200, 1000, 500)
    })
    X = data.drop(columns=["Total_Cost"])
    y = data["Total_Cost"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, "models/epc_cost_model.pkl")

# 📌 Train Smart Grid Load Forecasting Model
def train_smart_grid_model():
    print("Training Smart Grid Load Forecasting Model...")
    data = pd.DataFrame({
        "Hour": np.arange(0, 500),
        "Load_MW": np.random.uniform(500, 3000, 500),
        "Temperature": np.random.uniform(15, 45, 500),
        "Humidity": np.random.uniform(20, 90, 500)
    })
    X = data.drop(columns=["Load_MW"])
    y = data["Load_MW"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, "models/grid_forecasting.pkl")

# 📌 Train Procurement Optimization Model
def train_procurement_model():
    print("Training Procurement Optimization Model...")
    data = pd.DataFrame({
        "Material_Type": np.random.choice(["Steel", "Cement", "Cables"], 500),
        "Price_Per_Unit": np.random.uniform(100, 500, 500),
        "Logistics_Cost": np.random.uniform(50, 200, 500),
        "Total_Cost": np.random.uniform(150, 700, 500)
    })
    data = pd.get_dummies(data, columns=["Material_Type"])
    X = data.drop(columns=["Total_Cost"])
    y = data["Total_Cost"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, "models/procurement_ai.pkl")

# 📌 Train Predictive Maintenance Model
def train_maintenance_model():
    print("Training Predictive Maintenance Model...")
    data = pd.DataFrame({
        "Machine_Age_Years": np.random.randint(1, 20, 500),
        "Usage_Hours": np.random.randint(100, 10000, 500),
        "Vibration": np.random.uniform(0.1, 5.0, 500),
        "Temperature": np.random.uniform(30, 100, 500),
        "Failure_Probability": np.random.uniform(0, 1, 500)
    })
    X = data.drop(columns=["Failure_Probability"])
    y = data["Failure_Probability"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, "models/predictive_maintenance.pkl")

# 📌 Train Oil & Gas Monitoring Model
def train_oil_gas_model():
    print("Training Oil & Gas Monitoring Model...")
    data = pd.DataFrame({
        "Pipeline_Age_Years": np.random.randint(1, 50, 500),
        "Pressure_Drop": np.random.uniform(0.1, 10.0, 500),
        "Flow_Rate": np.random.uniform(100, 10000, 500),
        "Leak_Risk": np.random.uniform(0, 1, 500)
    })
    X = data.drop(columns=["Leak_Risk"])
    y = data["Leak_Risk"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, "models/oil_gas_monitoring.pkl")

# 📌 Train All Models
if __name__ == "__main__":
    train_epc_model()
    train_smart_grid_model()
    train_procurement_model()
    train_maintenance_model()
    train_oil_gas_model()
    print("🔥 All AI Models Trained Successfully! 🚀")
