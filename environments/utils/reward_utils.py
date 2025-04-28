import numpy as np
from typing import Tuple, List, Dict, Any, Union, Optional
from environments.config import map_height, map_width, rmax, normigfactor


def sim_map(map_matrix: np.ndarray, m: np.ndarray) -> float:
    count_free = 0
    final_count = 0
    
    for i in range(map_height):
        for j in range(map_width):
            if map_matrix[i, j] == 0:  # Free cell in ground truth
                count_free += 1
                if m[i, j] < 0.5:  # Correctly identified as free
                    final_count += 1
    
    return (final_count / count_free) if count_free > 0 else 0


def total_len(localx: List[float], localy: List[float]) -> float:
    # print("localx", localx)
    # print("localy", localy)
    # if not localx or not localy or len(localx) <= 1:
    #     return 0.0
        
    dist = 0
    for i in range(len(localx) - 1):
        dist += np.sqrt((localx[i] - localx[i+1])**2 + (localy[i] - localy[i+1])**2)
    
    return dist


def count_unknown_cells(m: np.ndarray) -> int:
    return np.sum(m == 0.5)


def calc_pher(path_x: List[float], path_y: List[float], pher_map: np.ndarray) -> float:

    # Add small epsilon to avoid division by zero
    epsilon = 1e-5
    pher_map_safe = pher_map + epsilon
    
    pherval = 0
    for k in range(len(path_x)):
        pherval += pher_map_safe[int(path_y[k])][int(path_x[k])]
    
    return pherval


def update_pheromone_map(
    pher_map: np.ndarray, 
    path_x: List[float], 
    path_y: List[float], 
    initial_pher_val: float, 
    map_matrix: np.ndarray, 
    robot_state: np.ndarray, 
    decay_rate: float = 0.8
) -> np.ndarray:

    height, width = pher_map.shape
    updated_map = pher_map.copy()
    
    for i in range(len(path_y)):
        y = path_y[i]
        x = path_x[i]
        
        for w in range(width):
            for h in range(height):
                rkc = np.sqrt((w - y)**2 + (h - x)**2)
                if rkc < rmax:
                    updated_map[w, h] += initial_pher_val * np.exp(-rkc / 5)
    
    initial_pher_val *= decay_rate
    
    inflated_map = map_matrix.copy()
    loc = np.argwhere(inflated_map > 0.5)
    for h, w in loc:
        h_min = max(0, h-1)
        h_max = min(height, h+1)
        w_min = max(0, w-1)
        w_max = min(width, w+1)
        updated_map[h_min:h_max, w_min:w_max] += initial_pher_val / 10
    
    epsilon = 1e-5
    updated_map += epsilon
    norm = np.linalg.norm(updated_map)
    if norm > 0:
        updated_map = updated_map / norm
    
    return updated_map


def reward_space(
    m: np.ndarray, 
    nextm: np.ndarray, 
    localx: List[float], 
    localy: List[float], 
    y_frontier: List[int], 
    x_frontier: List[int], 
    x: np.ndarray, 
    x_prev: np.ndarray, 
    pher_map: np.ndarray, 
    pher_condition: bool
) -> Tuple[float, float, float, float]:

    pv = calc_pher(localx, localy, pher_map)
    
    from environments.utils.geometry_utils import lennorm
    normcost = lennorm([localx[-1], localy[-1]], y_frontier, x_frontier) if len(localx) > 0 else 1.0
    path_cost = total_len(localx, localy) / normcost if normcost > 0 else 0
    
    ig = (count_unknown_cells(m) - count_unknown_cells(nextm)) / normigfactor
    reward = ig - 2 * path_cost
    if pher_condition:
        reward += -1.0 * pv
    if np.array_equal(x[0][0:2], x_prev[0][0:2]):
        reward = -1.0
    
    return reward, ig, path_cost, pv


def finishreward(m: np.ndarray, localx: List[float], localy: List[float], map_matrix: np.ndarray) -> Tuple[float, float, float]:
    total_path_len = total_len(localx, localy) / 200  # Normalize by 200
    similarity = sim_map(map_matrix, m)
    reward = 0.0
    
    return reward, total_path_len, similarity