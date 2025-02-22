from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load trained AI models
epc_model = joblib.load("models/epc_cost_model.pkl")
grid_model = joblib.load("models/grid_forecasting.pkl")
procurement_model = joblib.load("models/procurement_ai.pkl")
maintenance_model = joblib.load("models/predictive_maintenance.pkl")
oil_gas_model = joblib.load("models/oil_gas_monitoring.pkl")

app = FastAPI()

class Query(BaseModel):
    category: str
    question: str

def predict_epc_cost():
    return f"AI Predicted EPC Cost: ₹{epc_model.predict([[200, 150, 75]])[0]:.2f} crore."

def forecast_grid():
    return f"AI Forecasted Grid Load: {grid_model.predict([[12, 35, 60]])[0]:.2f} MW."

def optimize_procurement():
    return f"AI Optimized Procurement Cost: ₹{procurement_model.predict([[300, 80, 120]])[0]:.2f} per unit."

def predict_maintenance():
    return f"AI Predicted Failure Risk: {maintenance_model.predict([[10, 5000, 2.0, 75]])[0]:.2f} probability."

def monitor_oil_gas():
    return f"AI Predicted Leak Risk: {oil_gas_model.predict([[15, 5.0, 8000]])[0]:.2f} probability."

@app.post("/query/")
async def answer_question(query: Query):
    category = query.category.lower()

    if category == "epc":
        response = predict_epc_cost()
    elif category == "smart grid":
        response = forecast_grid()
    elif category == "procurement":
        response = optimize_procurement()
    elif category == "maintenance":
        response = predict_maintenance()
    elif category == "oil & gas":
        response = monitor_oil_gas()
    else:
        response = "Invalid category. Choose EPC, Smart Grid, Procurement, Maintenance, or Oil & Gas."

    return {"answer": response}
