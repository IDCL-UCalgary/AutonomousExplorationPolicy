import numpy as np

# Map related variables
angle_step = 0.008
meas_phi = np.arange(-np.pi, np.pi, angle_step)
rmax, alpha, beta = 12, 1, 0.008
map_width, map_height = 50, 50
map_size = (map_width, map_height)
num_obstacles = 10
obstacle_size = (4, 4)
min_distance = 10

def cells_in_lidar(lidar_range: int, map_width: int, map_height: int) -> int:
    i = int(map_height / 2)
    j = int(map_width / 2)
    count = 0
    for x in range(i - (lidar_range + 1), i + (lidar_range + 1)):
        for y in range(j - (lidar_range + 1), j + (lidar_range + 1)):
            count += 1
    return count

normigfactor = cells_in_lidar(rmax, map_width, map_height)
number_of_clusters = 10
action_size = number_of_clusters
