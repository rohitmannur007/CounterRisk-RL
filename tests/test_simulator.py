import unittest
import numpy as np
from src.simulator.credit_sim import CreditSimulator
from src.rl.bandit_optimizer import ThompsonSamplingPolicy

class TestSimulator(unittest.TestCase):
    def setUp(self):
        self.sim = CreditSimulator(n_simulations=100)
    
    def test_run_counterfactual(self):
        from src.policies.base_policy import BasePolicy
        class DummyPolicy(BasePolicy):
            def decide(self, X): return np.ones(len(X))
            def update(self, r): pass
        res = self.sim.run_counterfactual(DummyPolicy())
        self.assertIn("profit", res)
        self.assertGreater(res["default_rate"], 0)
    
    def test_rl_update(self):
        policy = ThompsonSamplingPolicy()
        policy.update(np.array([100, -500]), np.array([1, 2]))
        self.assertEqual(policy.alpha[1], 2.0)  # Success update
        self.assertEqual(policy.beta[2], 2.0)   # Failure update

if __name__ == "__main__":
    unittest.main()