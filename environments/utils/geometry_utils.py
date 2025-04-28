
import numpy as np
from typing import Tuple, List
from environments.config import map_height, map_width, meas_phi, angle_step, rmax


def move(action: int, coordinates: Tuple[int, int]) -> Tuple[int, int]:
    MOVES = [
        (0, 1),    # Right
        (0, -1),   # Left
        (1, 0),    # Up
        (-1, 0),   # Down
        (1, 1),    # Up-Right
        (-1, 1),   # Down-Right
        (1, -1),   # Up-Left
        (-1, -1),  # Down-Left
    ]
    
    return (coordinates[0] + MOVES[action][0], coordinates[1] + MOVES[action][1])


def check_free_or_occupied(map_matrix: np.ndarray, x: int, y: int) -> bool:
    if x <= 0 or x >= map_width or y >= map_height or y <= 0:
        return False
    elif map_matrix[y, x] >= 0.5:
        return False
    return True


def obstacle_inflation(map_matrix: np.ndarray) -> np.ndarray:
    inflated_map = map_matrix.copy()
    loc = np.argwhere(inflated_map > 0.5)

    for h, w in loc:
        h_min = max(0, h-2)
        h_max = min(map_height, h+3)
        w_min = max(0, w-2)
        w_max = min(map_width, w+3)
        inflated_map[h_min:h_max, w_min:w_max] = 1

    return inflated_map


def obstacle_inflation_narrow(map_matrix: np.ndarray) -> np.ndarray:
    inflated_map = map_matrix.copy()
    height, width = inflated_map.shape

    loc = np.argwhere(inflated_map > 0.5)

    for h, w in loc:
        h_min = max(0, h-1)
        h_max = min(height, h+1)
        w_min = max(0, w-1)
        w_max = min(width, w+1)
        inflated_map[h_min:h_max, w_min:w_max] = 1

    inflated_map[0, :] = 1
    inflated_map[height-1, :] = 1
    inflated_map[:, 0] = 1
    inflated_map[:, width-1] = 1

    return inflated_map


def action_to_goal_case3(action: float, map_matrix: np.ndarray, x: List[float]) -> Tuple[float, float, float, bool]:
    if action >= 0 and action <= 180:
        index = int((np.pi + np.deg2rad(action)) / angle_step)
    elif action > 180 and action <= 360:
        action -= 360
        index = int((np.pi + np.deg2rad(action)) / angle_step)

    final_x = x[1]
    final_y = x[0]
    done = False
    reward = 0
    
    for r in range(1, int(map_height * np.sqrt(2))):
        x_g = x[1] + (r) * np.cos(meas_phi[index])
        y_g = x[0] + (r) * np.sin(meas_phi[index])
        
        if x_g > 0 and x_g < map_width and y_g > 0 and y_g < map_height:
            if map_matrix[int(y_g)][int(x_g)] < 0.5:
                final_x = int(x_g)
                final_y = int(y_g)
            elif map_matrix[int(y_g)][int(x_g)] == 1:
                done = True
                reward = -3
                break
    
    return final_y, final_x, reward, done


def random_start(map_matrix: np.ndarray) -> List[int]:
    map_height, map_width = map_matrix.shape
    
    while True:
        x = np.random.randint(5, map_width - 5)
        y = np.random.randint(5, map_height - 5)
        
        if obstacle_inflation(map_matrix)[y, x] == 0:
            return [y, x]


def calc_measurements(map_matrix: np.ndarray, robot_state: List[float], rmax: int) -> Tuple[np.ndarray, np.ndarray]:
    from environments.config import action_size
    
    measured_phi = np.arange(0, 2*np.pi, np.deg2rad(360/action_size))
    height, width = np.shape(map_matrix)
    x, y, theta = robot_state
    meas_r = rmax * np.ones(measured_phi.shape)
    
    for i in range(len(measured_phi)):
        for r in range(1, rmax):
            # Finding the coordinate of each cell 
            xi = int((x + r * np.cos(theta + measured_phi[i])))
            yi = int((y + r * np.sin(theta + measured_phi[i])))
            
            if (xi <= 0 or xi >= height or yi <= 0 or yi >= width):
                meas_r[i] = r
                break
            elif map_matrix[int(xi), int(yi)] == 1:
                meas_r[i] = r
                break
                
    return meas_r, measured_phi


def connecting_line(x1: int, x2: int, y1: int, y2: int) -> Tuple[List[int], List[int]]:

    xlist = []
    ylist = []
    constat = rmax
    for i in range(constat + 1):
        u = i / constat
        x = int(x1 * u + x2 * (1 - u))
        y = int(y1 * u + y2 * (1 - u))
        xlist.append(x)
        ylist.append(y)
        
    return ylist, xlist

def lennorm(robotloc,y_frontier, x_frontier):
    cost = np.zeros([len(x_frontier)])
    for j in range(len(x_frontier)):
        cost[j] = np.sqrt((robotloc[0]-y_frontier[j])**2 + (robotloc[1]-x_frontier[j])**2)
    return np.sum(cost)