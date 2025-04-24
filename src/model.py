import pandas as pd
import joblib

# Load the model
model = joblib.load("model/random_forest_model.pkl")

# Load new data (example using X_test)
X_new = pd.read_csv("data/X_test_scaled.csv")

# Predict
predictions = model.predict(X_new)

# Show predictions
print("🔍 Predictions for new data:")
print(predictions[:10])
