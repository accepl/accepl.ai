data = pd.DataFrame({
    "Pipeline_Age_Years": np.random.randint(1, 50, 1000),
    "Pressure_Drop": np.random.uniform(0.1, 10.0, 1000),
    "Flow_Rate": np.random.uniform(100, 10000, 1000),
    "Leak_Risk": np.random.uniform(0, 1, 1000)
})

X = data.drop(columns=["Leak_Risk"])
y = data["Leak_Risk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/oil_gas_monitoring.pkl")
