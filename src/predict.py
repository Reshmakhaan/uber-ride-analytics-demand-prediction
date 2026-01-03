# src/predict.py

import os
import joblib
import numpy as np

# -----------------------------
# Load trained model
# -----------------------------
def load_model():
    """
    Load the trained Uber demand model from the models folder.
    """
    model_path = os.path.join(os.path.dirname(__file__), '../models/uber_demand_model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return joblib.load(model_path)

# -----------------------------
# Load label encoder for 'Base'
# -----------------------------
def load_encoder():
    """
    Load LabelEncoder used to encode 'Base' zones.
    """
    encoder_path = os.path.join(os.path.dirname(__file__), '../models/base_encoder.pkl')
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder file not found at {encoder_path}")
    return joblib.load(encoder_path)

# -----------------------------
# Predict ride demand
# -----------------------------
def predict_demand(model, hour, day_of_week, month, base, le):
    """
    Predict Uber ride demand given hour, day_of_week, month, and Base zone.

    Parameters:
    - model : trained ML model
    - hour : int (0-23)
    - day_of_week : int (0=Monday, 6=Sunday)
    - month : int (1-12)
    - base : str (zone code, e.g. 'B02512')
    - le : LabelEncoder for Base column

    Returns:
    - predicted number of rides (float)
    """
    # Convert Base zone to numeric using encoder
    try:
        base_encoded = le.transform([base])[0]
    except ValueError:
        # If base not seen during training, use a default or mean encoding
        base_encoded = 0

    # Prepare feature array (must match training order)
    X = np.array([[hour, day_of_week, month, base_encoded]])

    # Predict
    pred = model.predict(X)

    # Return numeric value
    return float(pred[0])
