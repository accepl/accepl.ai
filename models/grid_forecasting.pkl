data = pd.DataFrame({
    "Hour": np.arange(0, 1000),
    "Load_MW": np.random.uniform(500, 3000, 1000),
    "Temperature": np.random.uniform(15, 45, 1000),
    "Humidity": np.random.uniform(20, 90, 1000)
})

X = data.drop(columns=["Load_MW"])
y = data["Load_MW"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/grid_forecasting.pkl")
