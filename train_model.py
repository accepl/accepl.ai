import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # Example model, change as needed

# Load your dataset (adjust this according to your dataset)
data = pd.read_csv('data.csv')  # Assuming you have a 'data.csv' file for training
X_train = data.drop('target', axis=1)  # Features
y_train = data['target']  # Target column

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model using joblib
joblib.dump(model, 'model.joblib')

print("Model training completed and saved as 'model.joblib'")
