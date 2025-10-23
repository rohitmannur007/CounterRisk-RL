import matplotlib
matplotlib.use("Agg")
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from src.models.uplift_model import UpliftModel
from src.data_loader import load_german_credit_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CreditSimulator:
    """Counterfactual simulator for credit policies."""
    
    def __init__(self, n_simulations: int = 1000, risk_threshold: float = 0.05):
        self.n_simulations = n_simulations
        self.risk_threshold = risk_threshold  # Max portfolio default rate
        self.X, self.y = load_german_credit_data()
        # Simulate initial treatments (random policy for fitting)
        self.initial_treatment = np.random.binomial(1, 0.7, len(self.X))  # 70% approval
        self.uplift_model = UpliftModel()
        self.uplift_model.fit(self.X, self.y, self.initial_treatment)
    
    def run_counterfactual(self, policy) -> Dict[str, float]:
        """Run sim under policy, return metrics."""
        treatment, outcome, profit = self.uplift_model.simulate_counterfactual(self.X[:self.n_simulations], policy.decide)
        default_rate = np.mean(outcome)
        total_profit = np.sum(profit)
        feasible = default_rate <= self.risk_threshold
        avg_uplift = 0.0  # Dummy uplift for now; replace with np.mean(uplift_scores) if defined        print(f"DEBUG: default_rate={default_rate}, threshold={self.risk_threshold}, feasible={feasible}")
        

        logger.info(f"Sim results: Profit={total_profit:.2f}, Default Rate={default_rate:.3f}, Feasible={feasible}")
        return {
    "profit": total_profit,
    "default_rate": default_rate,
    "feasible": feasible,
    "uplift": avg_uplift
        }
    def backtest_policies(self, policies: List, n_episodes: int = 10) -> Dict:
        """Backtest multiple policies over episodes."""
        results = {}
        for policy_name, policy in policies.items():
            episode_profits = []
            for _ in range(n_episodes):
                    res = self.run_counterfactual(policy)
    episode_profits.append(res["profit"])

            
            results[policy_name] = {
                "mean_profit": np.mean(episode_profits),
                "std_profit": np.std(episode_profits),
                "feasible_rate": np.mean("feasible_rate": np.mean([r["feasible"] for r in [self.run_counterfactual(policy) for _ in range(n_episodes)]]))
            }
        self.plot_backtest(results)
        return results
    
    def plot_backtest(self, results: Dict):
        """Plot profit comparison."""
        policies = list(results.keys())
        mean_profits = [results[p]["mean_profit"] for p in policies]
        std_profits = [results[p]["std_profit"] for p in policies]
        
        plt.figure(figsize=(10, 6))
        plt.bar(policies, mean_profits, yerr=std_profits, capsize=5, alpha=0.7)
        plt.title("Policy Backtest: Mean Profit (with Uncertainty)")
        plt.ylabel("Total Profit ($)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("policy_eval.png")
        plt.show()
        logger.info("Backtest plot saved as policy_eval.png")