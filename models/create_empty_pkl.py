import joblib
import os

# Create models directory if not exists
os.makedirs("models", exist_ok=True)

# File names for the empty .pkl files
model_files = [
    "models/epc_cost_model.pkl",
    "models/grid_forecasting.pkl",
    "models/procurement_ai.pkl",
    "models/predictive_maintenance.pkl",
    "models/oil_gas_monitoring.pkl"
]

# Save empty dictionary in each .pkl file
for file in model_files:
    joblib.dump({}, file)
    print(f"✅ Created empty {file}")

print("🔥 All empty .pkl files created successfully!")
