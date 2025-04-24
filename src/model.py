from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def build_model(n_estimators=100, random_state=42):
    """
    Create and return a RandomForestRegressor model.
    """
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    return model

def save_model(model, path="model/random_forest_model.pkl"):
    """
    Save the trained model to the specified path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"💾 Model saved to: {path}")

def load_model(path="model/random_forest_model.pkl"):
    """
    Load a saved model from the specified path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    return joblib.load(path)
