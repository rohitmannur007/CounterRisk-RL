import logging
import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_german_credit_data(data_dir: str = "../data") -> Tuple[pd.DataFrame, np.ndarray]:
    """Load and preprocess German Credit Dataset. Returns X, y (0=good, 1=bad)."""
    data_path = os.path.join(data_dir, "german.data")
    if not os.path.exists(data_path):
        logger.warning("Dataset not found. Generating synthetic data.")
        return generate_synthetic_data(n_samples=1000)

    # Column names from UCI doc
    columns = [
        "status", "duration", "credit_history", "purpose", "credit_amount", "savings", "employment",
        "installment_rate", "status_sex", "other_debtors", "residence_duration", "property",
        "age", "installment_plans", "housing", "existing_credits", "job", "dependents",
        "telephone", "foreign_worker", "credit_risk"  # Target: 1=good, 2=bad
    ]
    
    # Load raw data (space-separated, no header)
    df = pd.read_csv(data_path, sep=" ", header=None, names=columns[:-1])  # Last is target, but file has 20 feats + target
    # Actually, file has 20 feats + target as last column
    df = pd.read_csv(data_path, sep=" ", header=None, names=columns)
    
    # Map target: 1=good (0), 2=bad (1)
    df["credit_risk"] = (df["credit_risk"] == 2).astype(int)
    
    # Encode categoricals
    categoricals = ["status", "credit_history", "purpose", "savings", "employment", "status_sex",
                    "other_debtors", "property", "installment_plans", "housing", "job",
                    "telephone", "foreign_worker"]
    le = LabelEncoder()
    for col in categoricals:
        df[col] = le.fit_transform(df[col].astype(str))
    
    # Numerical features
    numericals = ["duration", "credit_amount", "installment_rate", "residence_duration", "age",
                  "existing_credits", "dependents"]
    scaler = StandardScaler()
    df[numericals] = scaler.fit_transform(df[numericals])
    
    X = df.drop("credit_risk", axis=1).values
    y = df["credit_risk"].values
    
    logger.info(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features.")
    return X, y

def generate_synthetic_data(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback: Generate synthetic credit data (normal dist + binary target)."""
    np.random.seed(42)
    n_features = 20
    X = np.random.randn(n_samples, n_features)
    # Simulate target: higher features -> higher risk
    logit = np.dot(X, np.ones(n_features) * 0.5) + np.random.randn(n_samples) * 0.5
    y = (1 / (1 + np.exp(-logit)) > 0.7).astype(int)  # ~30% bad rate
    logger.info(f"Generated synthetic data: {n_samples} samples.")
    return X, y