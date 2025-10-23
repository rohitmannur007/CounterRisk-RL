import logging
from simulator.credit_sim import CreditSimulator
from src.rl.bandit_optimizer import ThompsonSamplingPolicy
from policies.base_policy import BasePolicy  # For dummy policy
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RandomPolicy(BasePolicy):
    """Baseline random policy."""
    def decide(self, X):
        return np.random.binomial(1, 0.5, len(X))
    def update(self, rewards, feasible):
        pass  # No update

def main():
    logger.info("Starting Counterfactual Credit Policy Simulator + RL Optimizer")
    
    # Init simulator
    sim = CreditSimulator(n_simulations=500, risk_threshold=0.30)
    
    # Policies
    policies = {
        "Random": RandomPolicy(),
        "RL_Optimized": ThompsonSamplingPolicy()
    }
    
    # Backtest
    results = sim.backtest_policies(policies)
    
    # RL Loop: Optimize over 20 episodes
    logger.info("Running RL optimization loop...")
    rl_policy = policies["RL_Optimized"]
    profits_over_time = []
    for episode in range(100):
        res = sim.run_counterfactual(rl_policy)
        treatment, outcome, profit = sim.uplift_model.simulate_counterfactual(sim.X[:sim.n_simulations], rl_policy.decide)
        arms_pulled = np.random.randint(0, rl_policy.n_arms, len(treatment))  # Proxy arms
        rl_policy.update(profit, arms_pulled, res["feasible"])
        profits_over_time.append(res["profit"])
        logger.info(f"Episode {episode+1}: Profit={res['profit']:.2f}")
    
    # Plot profit curve
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(profits_over_time, marker='o')
    plt.title("RL Policy Learning Curve: Cumulative Profit")
    plt.xlabel("Episode")
    plt.ylabel("Profit ($)")
    plt.grid(True)
    plt.savefig("profit_curve.png")
    plt.show()
    logger.info("RL plot saved as profit_curve.png")
    
    # Final metrics
    final_res = sim.run_counterfactual(rl_policy)
    logger.info(f"Final RL Policy: Profit={final_res['profit']:.2f}, Default Rate={final_res['default_rate']:.3f}")

if __name__ == "__main__":
    main()