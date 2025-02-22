from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import pickle
import numpy as np

# Create the FastAPI app
app = FastAPI()

# Serve static files (like index.html)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load pre-trained model (update path if needed)
with open('model.joblib', 'rb') as model_file:
    model = pickle.load(model_file)

# Example route for prediction
@app.get("/predict")
def predict(project_scope: str, material_cost: float, labor_cost: float):
    # Example prediction logic (replace with real feature engineering and prediction)
    features = np.array([material_cost, labor_cost]).reshape(1, -1)
    prediction = model.predict(features)
    return {"project_scope": project_scope, "predicted_cost": prediction[0]}

# Route to serve index.html
@app.get("/")
def read_index():
    return {"message": "Welcome to the AI-powered EPC app!"}
