import logging
import numpy as np
import pandas as pd
from typing import Tuple
import lightgbm as lgb
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UpliftModel:
    """Simple uplift model using two LightGBM models (treated/untreated) for counterfactuals.
    Approximates causal forest by difference in predictions.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.treated_model = None
        self.untreated_model = None
        self.feature_names = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, treatment: np.ndarray) -> None:
        """Fit on data with simulated treatment (approval=1)."""
        self.feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=self.feature_names)
        df["y"] = y
        df["treatment"] = treatment
        
        # Split treated/untreated
        treated = df[df["treatment"] == 1]
        untreated = df[df["treatment"] == 0]
        
        if len(treated) == 0 or len(untreated) == 0:
            raise ValueError("Need both treated and untreated samples.")
        
        X_t, y_t = treated[self.feature_names].values, treated["y"].values
        X_u, y_u = untreated[self.feature_names].values, untreated["y"].values
        
        # Train separate models
        self.treated_model = lgb.LGBMClassifier(random_state=self.random_state, n_estimators=50, verbose=-1)
        self.treated_model.fit(X_t, y_t)
        
        self.untreated_model = lgb.LGBMClassifier(random_state=self.random_state, n_estimators=50, verbose=-1)
        self.untreated_model.fit(X_u, y_u)
        
        logger.info("Uplift model fitted.")
    
    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        """Predict uplift: P(default|treated) - P(default|untreated). Negative = positive uplift (less default)."""
        if self.treated_model is None or self.untreated_model is None:
            raise ValueError("Model not fitted.")
        
        p_treated = self.treated_model.predict_proba(X)[:, 1]  # P(default|treated)
        p_untreated = self.untreated_model.predict_proba(X)[:, 1]  # P(default|untreated)
        uplift = p_treated - p_untreated  # Higher uplift = worse outcome from treatment
        return -uplift  # Flip to positive uplift = better outcome
    
    def simulate_counterfactual(self, X: np.ndarray, policy: callable) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate outcomes under policy (returns treatment, outcome, profit)."""
        treatment = policy(X)  # 0/1 approval
        uplift = self.predict_uplift(X)
        baseline_risk = self.untreated_model.predict_proba(X)[:, 1]
        risk = np.where(treatment == 1, baseline_risk + uplift, baseline_risk)
        outcome = (np.random.rand(len(X)) < risk).astype(int)  # 1=bad/default
        profit = np.where(treatment == 0, 0, np.where(outcome == 0, 1000, -500))  # Good: +1000 profit, Bad: -500 loss
        return treatment, outcome, profit