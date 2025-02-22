data = pd.DataFrame({
    "Machine_Age_Years": np.random.randint(1, 20, 1000),
    "Usage_Hours": np.random.randint(100, 10000, 1000),
    "Vibration": np.random.uniform(0.1, 5.0, 1000),
    "Temperature": np.random.uniform(30, 100, 1000),
    "Failure_Probability": np.random.uniform(0, 1, 1000)
})

X = data.drop(columns=["Failure_Probability"])
y = data["Failure_Probability"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/predictive_maintenance.pkl")
