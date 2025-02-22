import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

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

# Model to predict ROI and Tariff
model = LinearRegression()
model.fit(X, y)

def predict_roi_and_tariff(material_cost, labor_cost, time_to_complete):
    prediction = model.predict(np.array([[material_cost, labor_cost, time_to_complete]]))
    roi, tariff = prediction[0]
    return roi, tariff
