from fastapi import FastAPI
from pydantic import BaseModel
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
    question = query.question.lower()

    if category == "epc":
        response = epc.predict_cost(question)
    elif category == "smart grid":
        response = grid.forecast_load(question)
    elif category == "procurement":
        response = procurement.optimize_procurement(question)
    elif category == "maintenance":
        response = maintenance.predict_failures(question)
    elif category == "oil & gas":
        response = oil_gas.monitor_pipelines(question)
    else:
        response = "Invalid category. Please choose EPC, Smart Grid, Procurement, Maintenance, or Oil & Gas."

    return {"answer": response}
