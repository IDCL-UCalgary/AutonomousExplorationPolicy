import numpy as np

def cells_in_lidar(lidarrange,map_width, map_height):
    i = int(map_height/2)
    j = int(map_width/2)
    count = 0 
    for x in range(i-(lidarrange+1),i+(lidarrange+1)):
        for y in range(j-(lidarrange+1),j+(lidarrange+1)):
            count += 1 
    return count 


# map related variables 
# summarize all variables in one or two lines 
angle_step = 0.008
meas_phi = np.arange(-np.pi,np.pi,angle_step)
rmax , alpha , beta  = 12 , 1 , 0.008   
map_width , map_height = 50 , 50
map_size = (map_width,map_height)
num_obstacles = 10
obstacle_size = (4, 4)
min_distance = 10

# state, action, and reward related variables 
normigfactor = cells_in_lidar(rmax,map_width,map_height)
number_of_clusters = 10
action_size = number_of_clusters

# common variables between all DRL algorithms 
Trainig_Episodes_NO = 15000
maxep = 1
maxstep = 100 
gamma = 0.99 
lstm_time_step = 4


# default ppo related variables 
ppo_actor_lr = 1e-4 
ppo_entropy_coefficent = 0.001
ppo_clip_ratio = 0.2
ppo_batch_size = 6
ppo_n_epochs = 3 
ppo_lmbda = 0.95 
ppo_n_workers = 1


# default SAC related variables 
sac_lr = 1e-4
sac_batch_size = 8
sac_buffer_size = 10000
sac_tau = 0.001
sac_alpha = 0.1

# default D3QN related variables

D3QN_lr = 1e-4
D3QN_batch_size = 6
D3QN_buffer_size = 10000
D3QN_eps = 1
D3QN_eps_decay = 0.999
D3QN_eps_min = 0.01

# default DQN related variables
DQN_lr = 1e-4
DQN_batch_size = 6
DQN_buffer_size = 10000
DQN_eps = 1
DQN_eps_decay = 0.999
DQN_eps_min = 0.01



    