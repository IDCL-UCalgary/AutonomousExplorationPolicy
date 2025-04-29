from typing import Tuple, List, Dict, Any, Union, Optional, Callable
import numpy as np
from environments.utils.frontier_utils import frontier_space
from environments.utils.reward_utils import (
    reward_space, finishreward, sim_map, total_len, update_pheromone_map
)
from environments.utils.geometry_utils import action_to_goal_case3
from environments.config import (
    map_height, map_width, number_of_clusters, action_size, rmax
)


class ExplorationStrategy:
    
    def __init__(self, action_setting: int = 1, use_pheromones: bool = False):
        self.action_setting = action_setting
        self.use_pheromones = use_pheromones
        self.action_dim = action_size
        self.num_of_clusters = number_of_clusters
    
    def select_target(self, state: np.ndarray, action: int, robot_position: np.ndarray, 
                     map_data: np.ndarray, pheromone_map: Optional[np.ndarray] = None) -> Tuple[List[int], List[int], List[int]]:
        raise NotImplementedError("Subclasses must implement select_action method")
    
    def update_pheromones(self, pheromone_map: np.ndarray,
                         path_x: List[float], path_y: List[float],
                         initial_value: float, map_data: np.ndarray,
                         robot_position: np.ndarray, decay_rate: float = 0.8) -> np.ndarray:
        if self.use_pheromones:
            return update_pheromone_map(
                pheromone_map, path_x, path_y, 
                initial_value, map_data, robot_position, decay_rate
            )
        return pheromone_map


class RandomExplorationStrategy(ExplorationStrategy):    
    def select_target(self, state: np.ndarray, action: int, robot_position: np.ndarray, 
                     map_data: np.ndarray, pheromone_map: Optional[np.ndarray] = None) -> int:
        x_frontiers, y_frontiers = frontier_space(map_data, self.num_of_clusters, robot_position)
        goal_idx = np.random.randint(0, len(x_frontiers))
        goal = [int(y_frontiers[goal_idx]), int(x_frontiers[goal_idx]), 0]
        return goal , x_frontiers , y_frontiers


class RLExplorationStrategy(ExplorationStrategy):
    
    def __init__(self, action_setting: int = 1, use_pheromones: bool = False):
        super().__init__(action_setting, use_pheromones)
        # self.agent = agent
        # print("action setting", self.action_setting)
        # print("use pheromones", self.use_pheromones)
    
    def select_target(self, state: np.ndarray, action: int, robot_position: np.ndarray, 
                     map_data: np.ndarray, pheromone_map: Optional[np.ndarray] = None) -> Tuple[List[int], List[int], List[int]]:
        # if self.agent is None:
        #     raise ValueError("Agent not set for RLExplorationStrategy")
        
        # action = self.agent.act(state)
        if self.action_setting == 1:
            x_frontiers, y_frontiers = frontier_space(map_data, self.num_of_clusters, robot_position)
            distances = np.sqrt(
                (robot_position[0] - y_frontiers)**2 + 
                (robot_position[1] - x_frontiers)**2
            )
            sorted_indices = np.argsort(distances)
            goal_idx = sorted_indices[action]
            goal = [int(y_frontiers[goal_idx]), int(x_frontiers[goal_idx]), 0]
            
        elif self.action_setting == 2:
            x_frontiers, y_frontiers = frontier_space(map_data, self.num_of_clusters, robot_position)
            goal_idx = action
            goal = [int(y_frontiers[goal_idx]), int(x_frontiers[goal_idx]), 0]
            
        elif self.action_setting in (3, 4):
            from environments.utils.geometry_utils import obstacle_inflation, obstacle_inflation_narrow
            
            y_g, x_g, r, done = action_to_goal_case3(
                (action * 360 / action_size),
                obstacle_inflation(map_data) if self.action_setting == 3 else map_data,
                robot_position
            )
            goal = [int(y_g), int(x_g), 0]
        return goal , x_frontiers , y_frontiers


# class FrontierBasedExplorationStrategy(ExplorationStrategy):
#     # This class should be updated! not complete yet ; Content of nonlearning should be added but more structured :D 