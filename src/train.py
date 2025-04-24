import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import matplotlib.pyplot as plt


# Create the model directory if it doesn't exist
os.makedirs("../model", exist_ok=True)

DATA_DIR = "data"

MODEL_PATH = "model/random_forest_model.pkl"  # Update the path to match the created directory

# === Load Preprocessed Data ===
X_train = pd.read_csv(f"{DATA_DIR}/X_train_scaled.csv")
X_test = pd.read_csv(f"{DATA_DIR}/X_test_scaled.csv")
y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").values.ravel()
y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").values.ravel()

# === Train the Model ===
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Make Predictions ===
y_pred = model.predict(X_test)

# === Evaluate Model ===
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("✅ Model training complete.")
print(f"📉 Mean Squared Error (MSE): {mse:.2f}")
print(f"📈 R-squared (R²): {r2:.2f}")

# === Save Model ===
os.makedirs("model", exist_ok=True)  # Ensure the directory exists
with open(MODEL_PATH, 'wb') as f:
    joblib.dump(model, f)
print(f"💾 Model saved to: {MODEL_PATH}")


plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Cell Count")
plt.ylabel("Predicted Cell Count")
plt.title("Predicted vs. Actual Cell Counts")
plt.grid(True)
plt.show()

