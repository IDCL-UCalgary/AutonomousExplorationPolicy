from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List, Union, Optional
import numpy as np


class BaseAgent(ABC):
    def __init__(self, state_shape: Tuple, action_shape: Tuple, config: Dict[str, Any]):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.config = config
        self.is_training = True
    
    @abstractmethod
    def act(self, state: np.ndarray, deterministic: bool = False) -> Any:
        pass
    
    @abstractmethod
    def train(self, experiences: Any) -> Dict[str, float]:
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        pass
    
    def train_mode(self) -> None:
        self.is_training = True
    
    def eval_mode(self) -> None:
        self.is_training = False


class DiscreteAgent(BaseAgent):
    def __init__(self, state_shape: Tuple, action_dim: int, config: Dict[str, Any]):
        super().__init__(state_shape, (action_dim,), config)
        self.action_dim = action_dim
    
    def get_random_action(self) -> int:
        return np.random.randint(0, self.action_dim)


class ContinuousAgent(BaseAgent):
    
    def __init__(self, state_shape: Tuple, action_dim: int, action_bounds: Tuple[np.ndarray, np.ndarray], config: Dict[str, Any]):
        super().__init__(state_shape, (action_dim,), config)
        self.action_dim = action_dim
        self.action_low, self.action_high = action_bounds
    
    def get_random_action(self) -> np.ndarray:
        return np.random.uniform(self.action_low, self.action_high, size=self.action_dim)