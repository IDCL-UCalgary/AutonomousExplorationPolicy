import numpy as np
from typing import Tuple, List, Dict, Any, Union, Optional
from environments.mapping import map_update
from environments.utils.frontier_utils import frontier_space
from environments.utils.geometry_utils import action_to_goal_case3, random_start
from environments.config import (
    map_height, map_width, number_of_clusters, action_size, rmax, meas_phi, normigfactor
)
from environments.exploration import ExplorationStrategy, RandomExplorationStrategy
from environments.motion import generate_exploration_path
from environments.utils.reward_utils import (
    reward_space, finishreward, sim_map, total_len, update_pheromone_map
)


class ExplorationEnv:

    
    def __init__(self, map_matrix: np.ndarray, action_setting: int = 1, 
                 use_pheromones: bool = False, number_of_robots: int = 1,
                 exploration_strategy: Optional[ExplorationStrategy] = None):
        self.env_number = action_setting
        self.map_matrix = map_matrix
        self.map_height = map_height
        self.map_width = map_width
        self.use_pheromones = use_pheromones
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
        
        if self.use_pheromones:
            self.state_dim = (map_height, map_width, 2)
        else:
            self.state_dim = (map_height, map_width, 1)
        
        if exploration_strategy is None:
            self.exploration_strategy = RandomExplorationStrategy(
                action_setting=action_setting, 
                use_pheromones=use_pheromones
            )
        else:
            self.exploration_strategy = exploration_strategy

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:


        x = np.array([[self.yr, self.xr, 0]])
        robot_count = len(x)
        
        m = np.multiply(0.5, np.ones((self.map_height, self.map_width)))
        
        for i in range(robot_count):
            m = map_update(x[i], self.map_matrix, m)
        
        pher_map = np.zeros((map_height, map_width))
        
        localx = []
        localy = []
        
        x_frontiers, y_frontiers = frontier_space(m, self.num_of_clusters, x[0])
        
        if self.use_pheromones:
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
        
        return state, info

    def step(self, state:np.ndarray , action: int, x: np.ndarray = None, m: np.ndarray = None, 
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
        goal , x_frontiers , y_frontiers = self.exploration_strategy.select_target(state, action, x[0], m, pher_map if self.use_pheromones else None)
        
        dumy_var = [int(x[0][0]), int(x[0][1]), int(x[0][2])]

        # This part should be replaced by the trajectory_generation.py
        try:
            generated_path = generate_exploration_path(dumy_var, goal, m)
        except:
            try:
                generated_path = generate_exploration_path(dumy_var, goal, self.map_matrix)
            except:
                generated_path = []
        
        if len(generated_path) == 0:
            done = True
            info = {
                'robot_position': x,
                'map': m,
                'path_x': localx,
                'path_y': localy,
                'pheromone_map': pher_map,
                'information_gain': 0,
                'path_cost': 0,
                'pheromone_value': 0,
                'map_similarity': 0
            }
            
            if self.use_pheromones:
                state = np.zeros((map_height, map_width, 2))
                state[:, :, 0] = np.copy(m)
                state[x[0][0], x[0][1], 0] = 0.8
                state[:, :, 1] = np.copy(pher_map)
            else:
                state = np.copy(m)
                state[x[0][0], x[0][1]] = 0.8
                
            return state, 0, done, info
        
        old_map = np.copy(m)
        stepx = []
        stepy = []
        
        x_prev = np.copy(x)
        
        # This should be replaced by a trajectory tracking algorithm (given the dynamics of a robot)
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
            pher_map,  self.use_pheromones
        )
        reward += r
        
        if self.use_pheromones: # update pheromone map shoud be integrated into the mapping module --> Should be updated 
            pher_map = self.exploration_strategy.update_pheromones(
                pher_map, stepx, stepy, 100, m, x[0], decay_rate=0.8
            )
        
        if sim_map(self.map_matrix, m) > 0.95:
            done = True
        
        # State representation (To be Updated!)
        if self.use_pheromones:
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
                    state[int(yyy), int(xxx), 0] = 0.3
        
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
        
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        # To be implemented - following the standard OpenAI Gym Style
        pass