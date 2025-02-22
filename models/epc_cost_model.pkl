import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Create synthetic data
data = pd.DataFrame({
    "Project_Size_MW": np.random.randint(50, 500, 1000),
    "Material_Cost": np.random.uniform(100, 600, 1000),
    "Labor_Cost": np.random.uniform(50, 300, 1000),
    "Equipment_Cost": np.random.uniform(30, 200, 1000),
    "Total_Cost": np.random.uniform(200, 1000, 1000)
})

X = data.drop(columns=["Total_Cost"])
y = data["Total_Cost"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/epc_cost_model.pkl")
