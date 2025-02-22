data = pd.DataFrame({
    "Material_Type": np.random.choice(["Steel", "Cement", "Cables"], 1000),
    "Price_Per_Unit": np.random.uniform(100, 500, 1000),
    "Logistics_Cost": np.random.uniform(50, 200, 1000),
    "Total_Cost": np.random.uniform(150, 700, 1000)
})

# One-hot encoding for categorical variables
data = pd.get_dummies(data, columns=["Material_Type"])

X = data.drop(columns=["Total_Cost"])
y = data["Total_Cost"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/procurement_ai.pkl")
