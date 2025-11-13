# pipeline/preprocess.py

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# =============================
# Configuration of feature types
# =============================

CATEGORICAL_COLS = [
    "gender", "state", "kyc_mode", "ovd_type", 
    "interest_type"
]

BOOLEAN_COLS = [
    "pep_flag", "kfs_provided"
]

NUMERIC_COLS = [
    "age_years", "bureau_score", "monthly_income_inr",
    "existing_monthly_obligations_inr", "requested_amount_inr",
    "tenure_months", "interest_rate_annual_pct", "processing_fee_inr",
    "other_charges_inr", "apr_pct", "foir_total_obligations_pct",
    "property_value_inr", "ltv_ratio"
]

# =============================
# Load your scaler
# =============================

try:
    SCALER = joblib.load("models/scaler.pkl")
except FileNotFoundError:
    SCALER = None
    print("⚠️ Warning: Scaler not found. Data will not be scaled.")

# =============================
# Utility functions
# =============================

def _encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label-encode categorical features to numeric.
    Note: Uses local LabelEncoder; ideally you should load encoders from training if saved.
    """
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def preprocess(applicant_data: dict) -> np.ndarray:
    """
    Converts raw applicant dictionary into a model-ready numeric vector.
    Includes label encoding, boolean conversion, and scaling.
    """

    # Convert dict → DataFrame
    df = pd.DataFrame([applicant_data])

    # Fill missing numeric values
    df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(0)

    # Convert boolean → int
    for col in BOOLEAN_COLS:
        df[col] = df[col].astype(int)

    # Encode categorical
    df = _encode_categorical(df)

    # Select all relevant features
    final_cols = NUMERIC_COLS + CATEGORICAL_COLS + BOOLEAN_COLS
    X = df[final_cols].to_numpy()

    # Apply scaler if available
    if SCALER is not None:
        X = SCALER.transform(X)

    return X


if __name__ == "__main__":
    # Example test
    sample = {
        "age_years": 32,
        "gender": "Male",
        "state": "Maharashtra",
        "kyc_mode": "Video KYC",
        "ovd_type": "Aadhaar",
        "interest_type": "Fixed",
        "pep_flag": False,
        "kfs_provided": True,
        "bureau_score": 720,
        "monthly_income_inr": 55000,
        "existing_monthly_obligations_inr": 12000,
        "requested_amount_inr": 400000,
        "tenure_months": 36,
        "interest_rate_annual_pct": 10.5,
        "processing_fee_inr": 2000.0,
        "other_charges_inr": 500.0,
        "apr_pct": 12.5,
        "foir_total_obligations_pct": 45.0,
        "property_value_inr": 1000000,
        "ltv_ratio": 0.4
    }

    X = preprocess(sample)
    print("✅ Processed input shape:", X.shape)
    print("✅ First row (scaled):", X[0])



