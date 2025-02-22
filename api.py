# api.py
from fastapi import FastAPI
import requests
import pandas as pd
import numpy as np

# Initialize the FastAPI app
app = FastAPI()

# Sample root route
@app.get("/")
def read_root():
    return {"message": "Welcome to Accepl.AI API!"}

# Example route to fetch some data
@app.get("/data")
def get_data():
    # Example data processing
    data = pd.DataFrame({
        "x": np.random.randn(10),
        "y": np.random.randn(10),
    })
    return data.to_dict()

# Error handling route for when something goes wrong
@app.get("/error")
def cause_error():
    try:
        # Intentionally raising an error to test error handling
        1 / 0
    except Exception as e:
        return {"error": str(e)}

# Sample AI model endpoint
@app.get("/predict")
def predict():
    # Placeholder for your AI model prediction logic
    return {"prediction": "AI model output here"}

