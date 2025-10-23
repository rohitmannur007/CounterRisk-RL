import numpy as np
from typing import Callable
from abc import ABC, abstractmethod

class BasePolicy(ABC):
    """Abstract base for credit policies."""
    
    @abstractmethod
    def decide(self, X: np.ndarray) -> np.ndarray:
        """Return treatment vector (0/1: deny/approve)."""
        pass
    
    @abstractmethod
    def update(self, rewards: np.ndarray) -> None:
        """Update policy based on rewards (for RL)."""
        pass