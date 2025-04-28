import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Paths
MODEL_PATH = "model/random_forest_model.pkl"
DATA_DIR = "data"

# === Load Test Data ===
X_test = pd.read_csv(f"{DATA_DIR}/X_test_scaled.csv")
y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").values.ravel()

# === Load Trained Model ===
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from: {MODEL_PATH}")
except FileNotFoundError:
    print(f"Model file not found at: {MODEL_PATH}")
    exit()

# === Make Predictions ===
y_pred = model.predict(X_test)

# === Evaluate Model ===
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

# === Plot Results and Save to File ===
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Cell Count")
plt.ylabel("Predicted Cell Count")
plt.title("Predicted vs. Actual Cell Counts")
plt.grid(True)

# Save the plot to a file
PLOT_PATH = "plots/predicted_vs_actual.png"
plt.savefig(PLOT_PATH, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {PLOT_PATH}")

plt.show()