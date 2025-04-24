import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# === CONFIGURATION ===
DATA_DIR = "data"
MODEL_PATH = "model/random_forest_model.pkl"

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
import os
os.makedirs("model", exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"💾 Model saved to: {MODEL_PATH}")
