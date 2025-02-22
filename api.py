from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Initialize FastAPI
app = FastAPI()

# Define the input model
class ProjectDetails(BaseModel):
    material_cost_per_mw: float
    labor_cost_per_mw: float
    time_to_complete: float

# Example training data for cost estimation
data = pd.DataFrame({
    'material_cost_per_mw': [50, 60, 55, 65, 70],
    'labor_cost_per_mw': [20, 25, 22, 24, 30],
    'time_to_complete': [12, 14, 13, 16, 15],
    'average_tariff': [4.5, 5.0, 4.7, 5.2, 5.3],
    'roi': [12, 14, 13, 15, 16],
})

X = data[['material_cost_per_mw', 'labor_cost_per_mw', 'time_to_complete']]
y = data[['roi', 'average_tariff']]

# Train a simple Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Function to predict ROI and tariff
def predict_roi_and_tariff(material_cost, labor_cost, time_to_complete):
    prediction = model.predict(np.array([[material_cost, labor_cost, time_to_complete]]))
    roi, tariff = prediction[0]
    return roi, tariff

@app.get("/")
def read_root():
    return {"message": "Welcome to Accepl.AI API!"}

@app.post("/calculate-roi-tariff/")
def calculate_roi_tariff(project: ProjectDetails):
    # Use the model to predict ROI and Tariff
    roi, tariff = predict_roi_and_tariff(project.material_cost_per_mw, project.labor_cost_per_mw, project.time_to_complete)
    return {"roi": roi, "tariff": tariff}

@app.get("/current-tariff/")
def get_current_tariff():
    # Placeholder function to fetch current solar tariff
    return {"current_solar_tariff": 5.0}  # Static value for example

@app.get("/material-price/")
def get_material_price():
    # Placeholder function to fetch current material price
    return {"current_steel_price": 50}  # Static value for example
