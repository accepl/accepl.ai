from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

# Load trained AI models
models_path = "models/"
epc_model = joblib.load(os.path.join(models_path, "epc_cost_model.pkl"))
grid_model = joblib.load(os.path.join(models_path, "grid_forecasting.pkl"))
procurement_model = joblib.load(os.path.join(models_path, "procurement_ai.pkl"))
maintenance_model = joblib.load(os.path.join(models_path, "predictive_maintenance.pkl"))
oil_gas_model = joblib.load(os.path.join(models_path, "oil_gas_monitoring.pkl"))

app = FastAPI()

class Query(BaseModel):
    category: str
    question: str

@app.get("/")
async def root():
    return {"message": "AI EPC System is Live!"}

@app.get("/query/")
@app.post("/query/")
async def answer_question(query: Query = None):
    if not query:
        return {"error": "Please send a valid JSON request body."}

    category = query.category.lower()

    if category == "epc":
        response = f"Predicted EPC Cost: ₹{epc_model.predict([[200, 150, 75]])[0]:.2f} crore."
    elif category == "smart grid":
        response = f"Predicted Grid Load: {grid_model.predict([[12, 35, 60]])[0]:.2f} MW."
    elif category == "procurement":
        response = f"Optimized Procurement Cost: ₹{procurement_model.predict([[300, 80, 120]])[0]:.2f} per unit."
    elif category == "maintenance":
        response = f"Predicted Failure Risk: {maintenance_model.predict([[10, 5000, 2.0, 75]])[0]:.2f} probability."
    elif category == "oil & gas":
        response = f"Predicted Leak Risk: {oil_gas_model.predict([[15, 5.0, 8000]])[0]:.2f} probability."
    else:
        response = "Invalid category. Choose EPC, Smart Grid, Procurement, Maintenance, or Oil & Gas."

    return {"answer": response}
