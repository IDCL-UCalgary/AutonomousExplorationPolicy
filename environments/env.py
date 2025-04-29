# Moha Note : Get Exploration out of the sim environment; Prepare the infrastructure for the MARL
import numpy as np
from typing import Tuple, List, Dict, Any, Union, Optional
from environments.mapping import map_update
from environments.motion import generate_exploration_path
from environments.utils.frontier_utils import frontier_space
from environments.utils.reward_utils import (reward_space, finishreward, sim_map, total_len, update_pheromone_map)
from environments.config import (map_height, map_width, number_of_clusters, action_size,rmax, meas_phi, normigfactor)
from environments.utils.geometry_utils import action_to_goal_case3, random_start


class ExplorationEnv:
    
    def __init__(self, map_matrix: np.ndarray, action_setting: int = 1, 
                 pher_condition: bool = False, number_of_robots: int = 1):
        self.env_number = action_setting
        self.map_matrix = map_matrix
        self.map_height = map_height
        self.map_width = map_width
        self.pher_condition = pher_condition
        self.action_dim = action_size
        self.num_of_clusters = number_of_clusters
        self.number_of_robots = number_of_robots
        
        self.yr_vec, self.xr_vec = [], []
        for _ in range(number_of_robots):
            yr, xr = random_start(self.map_matrix)
            self.yr_vec.append(yr)
            self.xr_vec.append(xr)
            
        if self.number_of_robots == 1:
            self.yr, self.xr = self.yr_vec[0], self.xr_vec[0]
        
        if self.pher_condition:
            self.state_dim = (map_height, map_width, 2)
        else:
            self.state_dim = (map_height, map_width, 1)

    def reset(self) -> np.ndarray:
        
        yr, xr = self.yr, self.xr
        x0 = [yr, xr, 0]
        x = np.array([x0])
        robot_count = len(x)
        
        m = np.multiply(0.5, np.ones((self.map_height, self.map_width)))
        
        for i in range(robot_count):
            m = map_update(x[i], self.map_matrix, m)
        
        pher_map = np.zeros((map_height, map_width))
        localx = []
        localy = []
        x_frontiers, y_frontiers = frontier_space(m, self.num_of_clusters, x[0])
        
        if self.pher_condition:
            state = np.zeros((map_height, map_width, 2))
            state[:, :, 0] = np.copy(m)
            state[x[0][0], x[0][1], 0] = 0.8  
            
            for i in range(len(x_frontiers)):
                state[y_frontiers[i], x_frontiers[i], 0] = 0.3
                
            state[:, :, 1] = np.copy(pher_map)  
        else:
            state = np.copy(m)
            state[x[0][0], x[0][1]] = 0.8  
            for i in range(len(x_frontiers)):
                state[y_frontiers[i]][x_frontiers[i]] = 0.3
        

        self._x = x
        self._m = m
        self._localx = localx
        self._localy = localy
        self._pher_map = pher_map


        info = {
            'robot_position': x,
            'map': m,
            'path_x': localx,
            'path_y': localy,
            'pheromone_map': pher_map,
        }

        
        
        return state , info

    def step(self, action: int, x: np.ndarray = None, m: np.ndarray = None, 
             localx: List = None, localy: List = None, pher_map: np.ndarray = None) -> Tuple:
        if x is None:
            if not hasattr(self, '_x'):
                raise ValueError("Robot position 'x' not provided and no internal state exists")
            x = self._x
        else:
            self._x = x
        if m is None:
            if not hasattr(self, '_m'):
                raise ValueError("Map 'm' not provided and no internal state exists")
            m = self._m
        else:
            self._m = m
            
        if localx is None:
            if not hasattr(self, '_localx'):
                self._localx = []
            localx = self._localx
        else:
            self._localx = localx
            
        if localy is None:
            if not hasattr(self, '_localy'):
                self._localy = []
            localy = self._localy
        else:
            self._localy = localy
            
        if pher_map is None:
            if not hasattr(self, '_pher_map'):
                self._pher_map = np.zeros((map_height, map_width))
            pher_map = self._pher_map
        else:
            self._pher_map = pher_map
            
        done = False
        r = 0  
        
   
        if self.env_number == 1:
            x_frontiers, y_frontiers = frontier_space(m, self.num_of_clusters, x[0])
            distances = np.sqrt((x[0][0] - y_frontiers)**2 + (x[0][1] - x_frontiers)**2)
            sorted_indices = np.argsort(distances)
            goal_idx = sorted_indices[action]
            goal = [int(y_frontiers[goal_idx]), int(x_frontiers[goal_idx]), 0]
            
        elif self.env_number == 2:
            x_frontiers, y_frontiers = frontier_space(m, self.num_of_clusters, x[0])
            goal_idx = action
            goal = [int(y_frontiers[goal_idx]), int(x_frontiers[goal_idx]), 0]
            
        elif self.env_number in (3, 4):
            from environments.utils.geometry_utils import obstacle_inflation, obstacle_inflation_narrow
            x_frontiers, y_frontiers = [], []
            for ii in range(action_size):
                yyy, xxx, _, _ = action_to_goal_case3(
                    (ii * 360 / action_size), 
                    obstacle_inflation_narrow(m), 
                    x[0]
                )
                x_frontiers.append(xxx)
                y_frontiers.append(yyy)
            
            y_g, x_g, r, done = action_to_goal_case3(
                (action * 360 / action_size),
                obstacle_inflation(m) if self.env_number == 3 else m,
                x[0]
            )
            goal = [int(y_g), int(x_g), 0]
        
        dumy_var = [int(x[0][0]), int(x[0][1]), int(x[0][2])]
        
        try:
            generated_path = generate_exploration_path(dumy_var, goal, m)
        except:
            try:
                generated_path = generate_exploration_path(dumy_var, goal, self.map_matrix)
            except:
                generated_path = []
                done = True
        
        old_map = np.copy(m)
        stepx = []
        stepy = []
        
        x_prev = np.copy(x)
        
        for j in range(0, int(len(generated_path))):
            x[0] = np.array([generated_path[j][0], generated_path[j][1], 0])
            localx = np.append(localx, x[0][1])
            localy = np.append(localy, x[0][0])
            stepx = np.append(stepx, x[0][1])
            stepy = np.append(stepy, x[0][0])
            m = map_update(x[0], self.map_matrix, m)
            
            if self.map_matrix[int(generated_path[j][0])][int(generated_path[j][1])] >= 0.5:
                x[0] = np.array([generated_path[j-1][0], generated_path[j-1][1], 0])
                break
        
        if len(stepx) == 0:
            stepx = np.append(stepx, x[0][1])
            stepy = np.append(stepy, x[0][0])
            m = map_update(x[0], self.map_matrix, m)
        
        new_map = np.copy(m)
        
        reward, ig, cost, pv = reward_space(
            old_map, new_map, stepx, stepy, 
            y_frontiers, x_frontiers, x, x_prev,
            pher_map, self.pher_condition
        )
        reward += r
        
        pher_map = update_pheromone_map(
            pher_map, stepx, stepy, 100, m, x[0], decay_rate=0.8
        )
        
        if sim_map(self.map_matrix, m) > 0.95:
            done = True
        
        if self.pher_condition:
            state = np.zeros((map_height, map_width, 2))
            state[:, :, 0] = np.copy(m)
            state[x[0][0], x[0][1], 0] = 0.8  
            
            if self.env_number in (1, 2):
                for i in range(len(x_frontiers)):
                    state[y_frontiers[i], x_frontiers[i], 0] = 0.3
                state[:, :, 1] = np.copy(pher_map)
            
            elif self.env_number in (3, 4):
                for i in range(action_size):
                    yyy, xxx, _, _ = action_to_goal_case3(
                        (i * 360 / action_size), m, x[0]
                    )
                    state[int(yyy), int(xxx), 0] = 0.3
                state[:, :, 1] = np.copy(pher_map)
        
        else:
            state = np.copy(m)
            state[x[0][0], x[0][1]] = 0.8  
            
            if self.env_number in (1, 2):
                x_frontiers, y_frontiers = frontier_space(m, self.num_of_clusters, x[0])
                for i in range(len(x_frontiers)):
                    state[y_frontiers[i]][x_frontiers[i]] = 0.3
            
            elif self.env_number in (3, 4):
                for i in range(action_size):
                    yyy, xxx, _, _ = action_to_goal_case3(
                        (i * 360 / action_size), m, x[0]
                    )
                    state[int(yyy), int(xxx)] = 0.3
        

        self._x = x
        self._m = m
        self._localx = localx
        self._localy = localy
        self._pher_map = pher_map
        
        info = {
            'robot_position': x,
            'map': m,
            'path_x': localx,
            'path_y': localy,
            'pheromone_map': pher_map,
            'information_gain': ig,
            'path_cost': cost,
            'pheromone_value': pv,
            'map_similarity': sim_map(self.map_matrix, m)
        }
        
        return state, reward, done, info

    def compute_final_reward(self, episode_step: Optional[int] = None) -> Tuple[float, Dict]:
        if not hasattr(self, '_m') or not hasattr(self, '_localx') or not hasattr(self, '_localy'):
            raise ValueError("Environment state not initialized. Call reset() first.")
            
        reward, total_path_length, map_similarity = finishreward(
            self._m, self._localx, self._localy, self.map_matrix
        )
        
        info = {
            'total_path_length': total_path_length,
            'map_similarity': map_similarity,
            'steps': episode_step
        }
        
        return reward, info
        
    def _map_update(self, x, map_matrix, m):
        from environments.mapping import map_update
        return map_update(x, map_matrix, m)
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        # Following the standard OpenAI Gym Style! To be implemented
        pass 