from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import models.epc_cost_model as epc
import models.grid_forecasting as grid
import models.procurement_ai as procurement
import models.predictive_maintenance as maintenance
import models.oil_gas_monitoring as oil_gas

app = FastAPI()

class Query(BaseModel):
    category: str
    question: str

@app.post("/query/")
async def answer_question(query: Query):
    category = query.category.lower()

    if category == "epc":
        response = epc.predict_cost()
    elif category == "smart grid":
        response = grid.forecast_load()
    elif category == "procurement":
        response = procurement.optimize_procurement()
    elif category == "maintenance":
        response = maintenance.predict_failures()
    elif category == "oil & gas":
        response = oil_gas.monitor_pipelines()
    else:
        response = "Invalid category. Choose EPC, Smart Grid, Procurement, Maintenance, or Oil & Gas."

    return {"answer": response}
