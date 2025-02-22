import os
import joblib

# Create models directory if not exists
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

# Create empty .pkl files
model_files = [
    "epc_cost_model.pkl",
    "grid_forecasting.pkl",
    "procurement_ai.pkl",
    "predictive_maintenance.pkl",
    "oil_gas_monitoring.pkl"
]

for model in model_files:
    empty_model = None  # Empty placeholder
    file_path = os.path.join(models_dir, model)
    joblib.dump(empty_model, file_path)
    print(f"Created empty model: {file_path}")

print("✅ Empty .pkl files created successfully! 🚀")
