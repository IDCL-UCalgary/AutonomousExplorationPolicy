from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List, Union, Optional
import numpy as np


class BaseEnvironment(ABC):
    @abstractmethod
    def reset(self) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        pass
    
    @abstractmethod
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        pass
    
    @abstractmethod
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        pass
    
    @property
    @abstractmethod
    def state_shape(self) -> Tuple:
        pass
    
    @property
    @abstractmethod
    def action_shape(self) -> Tuple:
        pass
    
    @property
    def is_discrete_action(self) -> bool:
        return True
    
    def close(self) -> None:
        pass


class DiscreteActionEnvironment(BaseEnvironment):

    @property
    @abstractmethod
    def action_dim(self) -> int:
        pass
    
    @property
    def action_shape(self) -> Tuple[int]:
        return (self.action_dim,)
    
    @property
    def is_discrete_action(self) -> bool:
        return True


class ContinuousActionEnvironment(BaseEnvironment):
    
    @property
    @abstractmethod
    def action_dim(self) -> int:
        pass
    
    @property
    @abstractmethod
    def action_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        pass
    
    @property
    def action_shape(self) -> Tuple[int]:
        return (self.action_dim,)
    
    @property
    def is_discrete_action(self) -> bool:
        
        return False


class RobotExplorationEnvironment(DiscreteActionEnvironment):
    
    @abstractmethod
    def get_map(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_robot_pose(self) -> Tuple[float, float, float]:
        pass
    
    @abstractmethod
    def get_exploration_metrics(self) -> Dict[str, float]:
        pass