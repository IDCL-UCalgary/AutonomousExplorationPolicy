import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
# import matplotlib.pyplot as plt
from scipy.ndimage import label
from scipy.spatial.distance import cdist


from config import map_height, map_width, rmax  , normigfactor , angle_step, meas_phi , action_size


# def move(action, coordinates) -> tuple:
#         MOVES = [        
#                 (0, 1), (0, -1), (1, 0), (-1, 0),        
#                 (1, 1), (-1, 1), (1, -1), (-1, -1),        
#                 (0, 2), (0, -2), (2, 0), (-2, 0),        
#                 (2, 2), (-2, 2), (2, -2), (-2, -2),
#                 ]
#         return coordinates[0] + MOVES[action][0], coordinates[1] + MOVES[action][1]
def move(action, coordinates) -> tuple:
    
    MOVES = [
            (0, 1), (0, -1), (1, 0),  (-1, 0), 
            (1, 1),  (-1, 1), (1, -1), (-1, -1),
            ]
    # Apply the selected move to the current coordinates
    return (coordinates[0] + MOVES[action][0], coordinates[1] + MOVES[action][1])



def check_free_or_occupied(map , x, y):
        if x<=0 or x >=map_width or y >= map_height or y<=0 :
                return False
        elif map[y, x] >= 0.5:
                return False
        return True

def obstacle_inflation(map_matrix):
    inflated_map = map_matrix.copy()
    height = len(inflated_map)
    width = len(inflated_map[0])
    loc = np.argwhere(inflated_map > 0.5)

    for h, w in loc:
        inflated_map[h-2:h+3, w-2:w+3] = 1

    return inflated_map



def obstacle_inflation_narrow(map_matrix):
    inflated_map = map_matrix.copy()
    height, width = inflated_map.shape

    loc = np.argwhere(inflated_map > 0.5)
    for h, w in loc:
        inflated_map[h-1:h+1, w-1:w+1] = 1

    inflated_map[0, :] = 1  # Top border
    inflated_map[height-1, :] = 1  # Bottom border
    inflated_map[:, 0] = 1  # Left border
    inflated_map[:, width-1] = 1  # Right border

    return inflated_map


def sim_map(map_matrix,m):
  count_free = 0 
  final_count = 0 
  for i in range(map_height):
    for j in range(map_width):
      if map_matrix[i,j] == 0 : 
        count_free += 1
        if m[i,j] < 0.5 : 
          final_count += 1 
  return (final_count/count_free)




def frontier_detection(map_matrix):
    x_frontier = []
    y_frontier = []

    # Pad the map with zeros to avoid boundary checks
    padded_map = np.pad(map_matrix, ((1, 1), (1, 1)), mode='constant', constant_values=0)

    # Check the conditions using NumPy's vectorized operations
    mask = (map_matrix < 0.5) & (
        (padded_map[:-2, 1:-1] == 0.5) |
        (padded_map[1:-1, 2:] == 0.5) |
        (padded_map[2:, 1:-1] == 0.5) |
        (padded_map[1:-1, :-2] == 0.5)
    )

    # Find the coordinates of the frontier cells
    y_frontier, x_frontier = np.where(mask)

    final_x = []
    final_y = []

    for i in range(len(x_frontier)):
        if x_frontier[i] <=0 or x_frontier[i] >= map_width or y_frontier[i]<=0 or y_frontier[i]>= map_height:
            pass 
        else:
            final_x.append(x_frontier[i])
            final_y.append(y_frontier[i])

    final_x , final_y = filter_frontier_points(final_x,final_y,map_matrix)

    return final_x, final_y



def filter_frontier_points(x_frontier, y_frontier, map_matrix):
    # Label all connected components of free space
    labeled, _ = label(map_matrix == 0)

    # Find the label of the largest free space component
    component_sizes = np.bincount(labeled.flat)
    component_sizes[0] = 0  # Ignore the background label in component sizes
    largest_component = np.argmax(component_sizes)

    # Filter out frontier points that are not in the largest connected free space
    filtered_x_frontier = []
    filtered_y_frontier = []
    for x, y in zip(x_frontier, y_frontier):
        if labeled[y, x] == largest_component:
            filtered_x_frontier.append(x)
            filtered_y_frontier.append(y)

    return filtered_x_frontier, filtered_y_frontier



def clustering_filtering(xloc, yloc, m, num_cluster,x):
    X = np.array([xloc, yloc]).T
    x_frontier = []
    y_frontier = []
    model = KMeans(n_clusters=num_cluster)
    model.fit(X)
    yhat = model.predict(X)
    clusters = np.unique(yhat)
    for cluster in clusters:
        row_ix = np.where(yhat == cluster)[0]
        mean_x = np.mean(X[row_ix, 0])
        mean_y = np.mean(X[row_ix, 1])
        centroid = [mean_y, mean_x]
        # if m[int(mean_y), int(mean_x)] != 0:  # Adjust if in unknown or occupied space
        # mean_y , mean_x = find_nearest_free_space(centroid, m, x)
        # find nearest coreesponding frontier
        mean_y , mean_x = find_nearest_free_space([yloc,xloc] , centroid)
        x_frontier.append(int(mean_x))
        y_frontier.append(int(mean_y))
    return x_frontier, y_frontier

def find_nearest_free_space(frontiers, centroid):
    # robot_position = [x[0],x[1]]
    frontiers = np.column_stack((frontiers[0],frontiers[1]))
    # calculate distances from robot to all frontiers
    distances = np.linalg.norm(np.array(frontiers) - np.array([centroid[0],centroid[1]]), axis=1)
    closest_idx = np.argmin(distances)
    return frontiers[closest_idx][0], frontiers[closest_idx][1]






# def find_nearest_free_space(centroid, map_matrix):
#     # Find all free space coordinates
#     free_space_coords = np.argwhere(map_matrix == 0)

#     # Compute distances from centroid to all free spaces
#     distances = cdist([centroid], free_space_coords)

#     # Find the closest free space
#     closest_idx = np.argmin(distances)
#     return free_space_coords[closest_idx]

import numpy as np
from scipy.ndimage import label

# def find_nearest_free_space(centroid, map_matrix , x):
#     robot_position = [x[0],x[1]]
#     # Label all connected components of free space
#     structure = np.array([[1,1,1], [1,1,1], [1,1,1]])  # 8-connectivity
#     labeled, ncomponents = label(map_matrix == 0, structure)
    
#     # Find the label of the largest free space component
#     component_sizes = np.bincount(labeled.flat)
#     component_sizes[0] = 0  # Ignore the background label in component size
#     largest_component = np.argmax(component_sizes)

#     # largest_component = np.argmax(np.bincount(labeled.flat)[1:]) + 1

#     # Ensure centroid is in the largest connected component
#     centroid_label = labeled[int(centroid[0]), int(centroid[1])]
#     if centroid_label != largest_component or centroid == robot_position:
#         free_space_indices = np.argwhere(labeled == largest_component)

#         # Remove the robot's current position from the free space indices
#         robot_index = np.where((free_space_indices == robot_position).all(axis=1))
#         free_space_indices = np.delete(free_space_indices, robot_index, axis=0)
        
#         # Compute distances from centroid to all free spaces in the largest component
#         distances = np.linalg.norm(free_space_indices - np.array([centroid[0],centroid[1]]), axis=1)
        
#         # Find the closest free space within the largest component
#         closest_idx = np.argmin(distances)
#         # print("Closest free space is at index", free_space_indices[closest_idx])
#         # print("Centroid is at location", centroid[0], centroid[1])
#         return free_space_indices[closest_idx][0], free_space_indices[closest_idx][1]
#     return centroid[0], centroid[1]


def repeat_cluster(actual_clusters_x,actual_clusters_y,number_of_desired_clusters):
  n = number_of_desired_clusters - len(actual_clusters_x)
  x_frontiers = np.tile(actual_clusters_x,n+1)
  x_frontiers = x_frontiers[0:number_of_desired_clusters]
  y_frontiers = np.tile(actual_clusters_y,n+1)
  y_frontiers = y_frontiers[0:number_of_desired_clusters]
  return x_frontiers,y_frontiers


def frontier_space(m,number_of_cluster,x):
    #xloc , yloc = frontier_detection(obstacle_inflation(m))
    xloc , yloc = frontier_detection(m)
    if len(xloc) < number_of_cluster:
        x_frontier =    xloc
        y_frontier =    yloc
    else :
        x_frontier , y_frontier = clustering_filtering(xloc,yloc,m,number_of_cluster,x)
    return x_frontier , y_frontier

def frontier_space_nonlearning(m,number_of_cluster,x):
    x_frontier = []
    y_frontier = []
    # Pad the map with zeros to avoid boundary checks
    padded_map = np.pad(m, ((1, 1), (1, 1)), mode='constant', constant_values=0)

    # Check the conditions using NumPy's vectorized operations
    mask = (m < 0.5) & (
        (padded_map[:-2, 1:-1] == 0.5) |
        (padded_map[1:-1, 2:] == 0.5) |
        (padded_map[2:, 1:-1] == 0.5) |
        (padded_map[1:-1, :-2] == 0.5)
    )

    # Find the coordinates of the frontier cells
    y_frontier, x_frontier = np.where(mask)

    # import matplotlib.pyplot as plt
    # plt.imshow(np.subtract(1,m),cmap='gray',vmin=0,vmax=1,origin='lower')
    # plt.plot(x_frontier,y_frontier,'ro')
    # plt.show()

    x_before_filter =  []
    y_before_filter = []
    if len(x_frontier) > number_of_cluster:
        final_x = []
        final_y = []

        for i in range(len(x_frontier)):
            if x_frontier[i] <=0 or x_frontier[i] >= map_width or y_frontier[i]<=0 or y_frontier[i]>= map_height:
                pass 
            else:
                final_x.append(x_frontier[i])
                final_y.append(y_frontier[i])
        xloc , yloc = final_x , final_y
        X = np.array([xloc, yloc]).T
        x_frontier = []
        y_frontier = []

        
        model = KMeans(n_clusters=number_of_cluster)
        model.fit(X)
        yhat = model.predict(X)
        clusters = np.unique(yhat)

        
        for cluster in clusters:
            row_ix = np.where(yhat == cluster)[0]
            mean_x = np.mean(X[row_ix, 0])
            mean_y = np.mean(X[row_ix, 1])
            centroid = [mean_y, mean_x]
            x_before_filter.append(mean_x)
            y_before_filter.append(mean_y)
            mean_y , mean_x = find_nearest_free_space([yloc,xloc] , centroid)
            x_frontier.append(int(mean_x))
            y_frontier.append(int(mean_y))
    # plt.imshow(np.subtract(1,m),cmap='gray',vmin=0,vmax=1,origin='lower')
    # plt.plot(x_before_filter,y_before_filter,marker='o',color='purple',linestyle='None')
    # plt.plot(x_frontier,y_frontier,'bo')
    # plt.show()
    return x_frontier, y_frontier





def IG(m,y_frontier,x_frontier,actionindex):
    I = np.zeros([len(x_frontier)])
    for j in range(len(x_frontier)):
        count = 0 
        for xx in range(int(x_frontier[j])-(rmax+1),int(x_frontier[j])+(rmax+2)):
                for yy in range(int(y_frontier[j])-(rmax+1),int(y_frontier[j])+(rmax+2)):
                        if xx<0 or xx> map_width-1 or yy<0 or yy>map_height-1 :
                                pass 
                        else :
                                if ((xx-x_frontier[j])**2 + (yy-y_frontier[j])**2)<= rmax**2 : 
                                        if m[yy][xx] == 0.5:
                                                count += 1
        I[j] = count
    normi = I/np.linalg.norm(I)
    #print(normi)
    return normi[actionindex]


def cost(robotloc,y_frontier, x_frontier,actionindex):
    cost = np.zeros([len(x_frontier)])
    for j in range(len(x_frontier)):
        cost[j] = np.sqrt((robotloc[0]-y_frontier[j])**2 + (robotloc[1]-x_frontier[j])**2)
    normc = cost/np.linalg.norm(cost)
    return normc[actionindex]

def lennorm(robotloc,y_frontier, x_frontier):
    cost = np.zeros([len(x_frontier)])
    for j in range(len(x_frontier)):
        cost[j] = np.sqrt((robotloc[0]-y_frontier[j])**2 + (robotloc[1]-x_frontier[j])**2)
    return np.sum(cost)




def reward_space(m,nextm,localx,localy,y_frontier,x_frontier,x,x_prev,pher_map,pher_conditoin):
    pv = calc_pher(localx,localy,pher_map) 
    normcost = lennorm([localx[-1],localy[-1]],y_frontier,x_frontier)
    cost = total_len(localx,localy)/normcost
    ig = (count_unkonwn_cells(m)-count_unkonwn_cells(nextm))/normigfactor
    reward = ig - 2*cost
    if pher_conditoin == True : 
        reward += -1.0*pv
    if x[0][0] == x_prev[0][0] and x[0][1] == x_prev[0][1]:
        # print("stuck! :/")
        reward = -1.0
    # print("ig is ",ig)
    # print("cost is ",cost)

    return reward , ig, cost, pv 




def finishreward(m,localx,localy,map_matrix):
    totalen = total_len(localx,localy)/(200) # 200 is hard coeded value for normalization however it should be based on the map size
    sim = sim_map(map_matrix,m)
    # reward = 1*(1.25-totalen) + 0.6*sim

    # reward = sim 
    reward = 0 
    # if sim > 0.9 :
    #     reward += 1

    return reward , totalen , sim



def total_len(localx,localy):
    dist = 0
    for i in range(len(localx)-1):
        dist += np.sqrt((localx[i]-localx[i+1])**2 + (localy[i]-localy[i+1])**2)

    return dist


def random_start(map):
    map_width = map.shape[1]
    map_height =map.shape[0]
    while True:
        x = np.random.randint(5,map_width-5)
        y = np.random.randint(5,map_height-5)
        if obstacle_inflation(map)[y, x] == 0 :
            return [y,x]
        

def count_unkonwn_cells(m):
    count_unkown = 0
    map_height = m.shape[0]
    map_width = m.shape[1]
    for i in range(map_height):
        for j in range(map_width):
            if m[i,j] == 0.5 : 
                count_unkown += 1
    return count_unkown


def calc_heading_error(theta_index,agent_state,state_frontier,action_size):
    yf,xf = state_frontier
    ya,xa = agent_state[0],agent_state[1]

    theta = theta_index * (2 * np.pi / action_size)
    if np.arctan2((yf-ya),(xf-xa)) < 0:
        return abs(2*np.pi + np.arctan2((yf-ya),(xf-xa)) - theta)
    else:
        return (abs(np.arctan2((yf-ya),(xf-xa)) - theta))


def select_nearest_frontier(theta_index,agent_state,x_frontier,y_frontier,action_size):
    best_frontier = [y_frontier[0],x_frontier[0]]
    best_heading_error = calc_heading_error(theta_index,agent_state,[y_frontier[0],x_frontier[0]],action_size)
    for i in range(len(x_frontier)):
        
        heading_error = calc_heading_error(theta_index,agent_state,[y_frontier[i],x_frontier[i]],action_size)
        if heading_error < best_heading_error:
            best_heading_error = heading_error
            best_frontier = [y_frontier[i],x_frontier[i]]
    return best_frontier


def calc_dist(cell,robot_state):
    return(np.sqrt((cell[0]-robot_state[0])**2+(cell[1]-robot_state[1])**2))


def update_pheromone_map(pher_map,path_x,path_y,initial_pher_val,map_matrix,robot_state,decay_rate=0.8):
    height = pher_map.shape[0]
    width = pher_map.shape[1]
    for i in range(len(path_y)):
        #print(len(path_y))
        #print(len(path_x))
        y = path_y[i]
        x = path_x[i]
        for w in range(width):
            for h in range(height):
                rkc = calc_dist([w,h],[y,x])
                #print(rkc)
                # if rkc <= 4*rmax :
                if rkc < rmax : 
                    pher_map[w][h] += initial_pher_val*np.exp(-rkc/5)
                    # pher_map[w][h] += initial_pher_val*np.exp(-rkc/10)
    initial_pher_val *= decay_rate
    inflated_map = map_matrix.copy()
    height, width = inflated_map.shape

    # Inflate the obstacles
    loc = np.argwhere(inflated_map > 0.5)
    for h, w in loc:
        pher_map[h-1:h+1, w-1:w+1] += initial_pher_val/10

    # Inflate the border
    # pher_map[0, :] += initial_pher_val  # Top border
    # pher_map[height-1, :] += initial_pher_val  # Bottom border
    # pher_map[:, 0] += initial_pher_val # Left border
    # pher_map[:, width-1] += initial_pher_val  # Right border

    # pher_map[robot_state[0],robot_state[1]] += initial_pher_val/2

    # x_frontiers ,  y_frontiers = frontier_detection(obstacle_inflation_narrow(map_matrix))
    # for i in range(len(x_frontiers)):
    #     pher_map[y_frontiers[i],x_frontiers[i]] = 0
    epsilon = 1e-5
    pher_map += epsilon
    pher_map = pher_map/np.linalg.norm(pher_map)
    return pher_map

def calc_pher(path_x,path_y,pher_map):
    #pherlist = np.ones([number_of_robots])*0.001
    # pherval = 0
    epsilon = 1e-5
    pher_map += epsilon
    pherval = 0 
    # #for i in range(number_of_robots):
    for k in range(len(path_x)):
        #print("pathy k is ",path_y[k])
        #print("pathx k is ",path_x[k])
        #print(pher_map.shape)
        pherval += pher_map[int(path_y[k])][int(path_x[k])]
    
    # pherval_prev = pher_map[int(path_y[-1])][int(path_x[-1])]

    return pherval

def sig_point(gp):
    ysigp = []
    xsigp = []
    if len(gp) > 0:
        ysigp.append(gp[0][0])
        xsigp.append(gp[0][1])

    for i in range(1,len(gp)-1):
        if (gp[i+1][0]- 2*gp[i][0]+ gp[i-1][0] != 0) or (gp[i+1][1]- 2*gp[i][1]+ gp[i-1][1] != 0):
            ysigp.append(gp[i][0])
            xsigp.append(gp[i][1])
    
    if len(gp) > 0:
        ysigp.append(gp[-1][0])
        xsigp.append(gp[-1][1])
    newgp = []
    for j in range(len(xsigp)):
        newgp.append([ysigp[j],xsigp[j]])
        #if np.sqrt((ysigp[j]-ysigp[j+1])**2 + (xsigp[j]-xsigp[j+1])**2) > 2 :
            #newgp.append([ysigp[j],xsigp[j]])

    #newgp.append([ysigp[-1],xsigp[-1]])
    return newgp



def calc_measurements(map_matrix,robot_state,rmax):
    measured_phi = np.arange(0,2*np.pi,np.deg2rad(360/action_size))
    (height,width) = np.shape(map_matrix)
    x = robot_state[0]
    y = robot_state[1]
    theta = robot_state[2]
    meas_r = rmax*np.ones(measured_phi.shape)
    for i in range(len(measured_phi)):
        for r in range(1,rmax):
            # finding the coordinate of each cell 
            xi = int((x+r*np.cos(theta+measured_phi[i])))
            yi = int((y+r*np.sin(theta+measured_phi[i])))
            if (xi <= 0 or xi>= height or yi<=0 or yi>= width):
                meas_r[i] = r 
                break
            elif map_matrix[int((xi)),int((yi))] == 1:
                meas_r[i] = r
                break 
    return meas_r , measured_phi



# def action_to_goal(action,meas_r,x,measured_phi):
#     # print(action)
#     # if action >=0 and action< 180:
#     #     index = int((np.pi+np.deg2rad(action))/np.deg2rad(360/action_size))
#     # elif action >= 180 and action<= 360:
#     #     action -= 360
#     #     index = int((np.pi+np.deg2rad(action))/np.deg2rad(360/action_size))
    
#     index = int((np.deg2rad(action))/np.deg2rad(360/action_size))
#     # print("index is ",index)
#     # print("Corresponding heading is",np.rad2deg(meas_phi[index]))
    
#     x_g = x[1] + (meas_r[index])*np.sin(measured_phi[index])
#     y_g = x[0] + (meas_r[index])*np.cos(measured_phi[index])

#     if x_g >= map_width :
#         x_g = map_width - 1 
#     if y_g >= map_height :
#         y_g = map_height - 1
#     if x_g < 0 :
#         x_g = 1 
#     if y_g < 0 :
#         y_g = 1 

#     return y_g , x_g 


def action_to_goal_case3(action,map_matrix,x):
    if action >=0 and action<= 180:
        index = int((np.pi+np.deg2rad(action))/angle_step)
    elif action > 180 and action<= 360:
        action -= 360
        index = int((np.pi+np.deg2rad(action))/angle_step)

    final_x = x[1]
    final_y = x[0]
    done = False 
    reward = 0 
    for r in range(1,int(map_height*np.sqrt(2))):
        x_g = x[1] + (r)*np.cos(meas_phi[index])
        y_g = x[0] + (r)*np.sin(meas_phi[index])
        if x_g >0 and x_g< map_width and y_g>0 and y_g< map_height:
            if map_matrix[int(y_g)][int(x_g)]<0.5 :
                final_x = int(x_g)
                final_y = int(y_g)
            elif map_matrix[int(y_g)][int(x_g)]==1: 
                done = True
                reward = - 3
                break
    return final_y , final_x , reward , done

def draw_range (map_matrix,x,y,meas_phi,rmax):
  (height,width) = np.shape(map_matrix)
  xlist = []
  ylist = []
  meas_r = rmax*np.ones(meas_phi.shape)
  for i in range(len(meas_phi)):
    for r in range(1,rmax+1):
      # finding the coordinate of each cell 
      xi = int(round(x+r*np.cos(meas_phi[i])))
      yi = int(round(y+r*np.sin(meas_phi[i])))

      if (xi <= 0 or xi>= height-1 or yi<=0 or yi>= width-1):
        break
      elif map_matrix[int(round(yi)),int(round(xi))] == 1:
        break
      xlist.append(xi)
      ylist.append(yi)
  return xlist,ylist


def connecting_line(x1,x2,y1,y2):
    xlist= []
    ylist= []
    constat = rmax
    for i in range(constat+1):
        u = i/constat
        x = int(x1*u + x2*(1-u))
        y= int(y1*u + y2*(1-u))
        xlist.append(x)
        ylist.append(y)
    return ylist,xlist
def calc_normpher_nonlearning(x,x_frontier,y_frontier,number_of_robots,pher_map):
    pherlist = np.ones([number_of_robots,len(x_frontier)])*0.001
    for i in range(number_of_robots):
        for j in range(len(x_frontier)):
            pathy,pathx = connecting_line(x[i][1],x_frontier[j],x[i][0],y_frontier[j])
            for k in range(len(pathy)):
                pherlist[i][j] += pher_map[pathy[k]][pathx[k]]
    normpher = (pherlist/np.linalg.norm(pherlist))
    return normpher

def find_max_len(myarr):
    mymax = 0 
    # print("myarr is ",myarr)
    for i in range(len(myarr)):
        if len(myarr[i][0])> mymax : 
            mymax = len(myarr[i][0])
            index = i
    return index 


def generate_animation(map_for_animation, path_x_for_animation, path_y_for_animation, output_filename,Robot_Number, fps=100):
    from matplotlib import pyplot as plt
    from matplotlib.animation import PillowWriter   
    
    writer = PillowWriter(fps=fps)

    fig, ax = plt.subplots()
    color_list = ['red', 'green', 'blue', 'orange']

    number = Robot_Number 
    index = find_max_len(path_x_for_animation)  # Assuming find_max_len function finds the maximum length among paths

    with writer.saving(fig, output_filename, 100):
        for j in range(len(map_for_animation[index][0])):  # Iterate up to the maximum length
            ax.clear()
            for i in range(number):
                if j < len(map_for_animation[i][0]):  # Ensure not to go beyond the available data
                    ax.imshow(np.subtract(1, map_for_animation[i][0][j]), cmap='gray', origin='lower', vmin=0.0, vmax=1.0)
                    ax.plot(path_x_for_animation[i][0][:j], path_y_for_animation[i][0][:j], '-', color=color_list[i], linewidth=2)  # Plot the path taken by each robot up to current frame
                    ax.plot(path_x_for_animation[i][0][j], path_y_for_animation[i][0][j], 'o', color=color_list[i], linewidth=2)
                else: 
                    ax.imshow(np.subtract(1, map_for_animation[i][0][-1]), cmap='gray', origin='lower', vmin=0.0, vmax=1.0)
                    ax.plot(path_x_for_animation[i][0][0:-1], path_y_for_animation[i][0][0:-1], color=color_list[i], linewidth=2)
                    ax.plot(path_x_for_animation[i][0][-1], path_y_for_animation[i][0][-1], 'o', color=color_list[i], linewidth=2)
            writer.grab_frame()