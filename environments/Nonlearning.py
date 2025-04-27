import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from config import  map_height, map_width  , rmax , number_of_clusters, maxstep
from mapping import map_update 
from motion import generate_exploration_path
from utils import ( frontier_space, reward_space, sim_map,total_len, calc_normpher_nonlearning , update_pheromone_map,random_start
                   ,frontier_space_nonlearning)
import random
import math

import matplotlib.pyplot as plt


class RRTGraph:
    # random.seed(10)
    def __init__(self, start,map_img):
        (x, y) = start
        self.start = start
        self.goalFlag = False
        self.maph = map_height # edit these two lines to fit this code for every map
        self.mapw = map_width
        self.x = []
        self.y = []
        self.parent = []
        # initialize the tree
        self.x.append(x)
        self.y.append(y)
        self.parent.append(0)
        # the obstacles
        # path
        self.goalstate = None
        self.path = []
        self.map_img = map_img

    def add_node(self, n, x, y):
        self.x.insert(n,x)
        self.y.append(y)

    def remove_node(self, n):
        self.x.pop(n)
        self.y.pop(n)

    def add_edge(self, parent, child):
        self.parent.insert(child, parent)

    def remove_edge(self, n):
        self.parent.pop(n)

    def number_of_nodes(self):
        return len(self.x)

    def distance(self, n1, n2):
        (x1, y1) = (self.x[n1], self.y[n1])
        (x2, y2) = (self.x[n2], self.y[n2])
        px = (float(x1) - float(x2)) ** 2
        py = (float(y1) - float(y2)) ** 2
        return (px + py) ** (0.5)

    def sample_envir(self):
        x = int(random.uniform(0, self.mapw))
        y = int(random.uniform(0, self.maph))
        return x, y

    def nearest(self, n):
        dmin = self.distance(0, n)
        nnear = 0
        for i in range(0, n):
            if self.distance(i, n) < dmin:
                dmin = self.distance(i, n)
                nnear = i
        return nnear

    def isFree(self):
        n = self.number_of_nodes() - 1
        (x, y) = (self.x[n], self.y[n])
        if x <= 0 or x >= map_width or y >= map_height or y <= 0:
            return False
        elif self.map_img[y,x]> 0.5 :
            return False
        return True

    def crossObstacle(self, x1, x2, y1, y2):
        xlist = []
        ylist = []
        for i in range(0,2001):
            u = i/2000
            x = x1*u + x2*(1-u)
            y = y1*u + y2*(1-u)
            if self.map_img[int(np.round(y)), int(np.round(x))]> 0.5 :
                return True
            xlist.append(x)
            ylist.append(y)
        return False

    def connect(self, n1, n2):
        (x1, y1) = (self.x[n1], self.y[n1])
        (x2, y2) = (self.x[n2], self.y[n2])
        if self.crossObstacle(x1, x2, y1, y2):
            self.remove_node(n2)
            self.goalFlag = False
        else:
            self.add_edge(n1, n2)
            return True

    def step(self, nnear, nrand, dmax=5):
        d = self.distance(nnear, nrand)
        if d > dmax:
            (xnear, ynear) = (self.x[nnear], self.y[nnear])
            (xrand, yrand) = (self.x[nrand], self.y[nrand])
            (px, py) = (xrand - xnear, yrand - ynear)
            theta = math.atan2(py, px)
            (x, y) = (int(xnear + dmax * math.cos(theta)),
                      int(ynear + dmax * math.sin(theta)))
            self.remove_node(nrand)
            if self.map_img[y][x] == 0.5 : 
              self.add_node(nrand,x,y)
              self.goalstate = nrand
              self.goalFlag = True
            else:
                self.add_node(nrand, x, y)

    def bias(self, ngoal):
        n = self.number_of_nodes()
        self.add_node(n, ngoal[0], ngoal[1])
        nnear = self.nearest(n)
        self.step(nnear, n)
        self.connect(nnear, n)
        return self.x, self.y, self.parent

    def expand(self):
        n = self.number_of_nodes()
        x, y = self.sample_envir()
        self.add_node(n, x, y)
        if self.isFree():
            nnearest = self.nearest(n)
            self.step(nnearest, n)
            self.connect(nnearest, n)
        else :
            self.remove_node(n)
        return self.x, self.y, self.parent

    def path_to_goal(self):
        if self.goalFlag:
            self.path = []
            self.path.append(self.goalstate)
            newpos = self.parent[self.goalstate]
            while (newpos != 0):
                self.path.append(newpos)
                newpos = self.parent[newpos]
            self.path.append(0)
        
        return self.goalFlag
    def getPathCoords(self):
        pathCoords = []
        for node in self.path:
            x, y = (self.x[node], self.y[node])
            pathCoords.append((x, y))
        return pathCoords
    

def bids_for_task_allocator(x_frontier,y_frontier,number_of_robots,x,m,localmaps):
    T = np.zeros([number_of_robots,len(x_frontier)])
    I = np.zeros([number_of_robots,len(x_frontier)])
    check = np.ones([number_of_robots,len(x_frontier)])*0.0001
    pheromnelist = np.zeros(len(x_frontier))

    for i in range(number_of_robots):
        T[i] = np.sqrt((x[i][0]-y_frontier)**2 +(x[i][1]-x_frontier)**2)
        

        for j in range(len(x_frontier)):
            count = 0 
            for xx in range(int(x_frontier[j])-(rmax),int(x_frontier[j])+(rmax)):
                for yy in range(int(y_frontier[j])-(rmax),int(y_frontier[j])+(rmax)):
                    if 0 <= xx < map_width and 0 <= yy < map_height:
                        if ((xx - x_frontier[j]) ** 2 + (yy - y_frontier[j]) ** 2) <= rmax ** 2:
                            if m[yy][xx] == 0.5:
                                count += 1
            I[i][j] = count
        
        for k in range(len(x_frontier)):
            # genereate a path between the current robot and the frontier point. if there could be a path then the check value will be 0. 
            #otherwise it will be 1e6
            try:
                generate_exploration_path([int(x[i][0]),int(x[i][1]),0],[y_frontier[k],x_frontier[k]],m)
            except:
                check[i][k] = 10**6

            # if localmaps[i][y_frontier[k]][x_frontier[k]] == 0.5 :
            #     check[i][k] = 1e6
            
    
    normi = I/np.linalg.norm(I)
    normt = T/np.linalg.norm(T)
    normcheck = check 
    # normcheck = check/np.linalg.norm(check)
    return normi,normt,normcheck



def task_allocator(x_frontier,y_frontier,RobotNO,normi,normt,normcheck,normpher,alpha=0.0 ,betta=1.5,gamma=1,lamb = 1):
    frontier_index = np.zeros(RobotNO,int)
    for i in range(RobotNO):
        p = alpha*normi - betta*normt - gamma*normcheck - lamb*normpher
        frontier_index[i] = np.where(p[i] == np.amax(p[i]))[0][0]
        for j in range(RobotNO-(i+1)):
            for k in range(len(x_frontier)):
                if ((x_frontier[frontier_index[i]]-x_frontier[k])**2 + (y_frontier[frontier_index[i]]-y_frontier[k])**2)<=(rmax)**2:
                    normt[j+i+1][k] *= 10
    return frontier_index


def nonlearningapproach(test_env , map_matrix ,RobotNO, alpha = 0.8 , betta=0.2, gamma=0, lamb = 0.0):
    hybrid = False
    term_criteria = 0.95
    (total_path_len , time_froniter, all_rewards , all_ig , 
     all_pv , all_cost , all_sim , all_count , Topo_length , 
     plot_sim_fr , plot_len_fr, action_hist) = [],[],[],[],[],[],[],[],[],[],[],[]
    
    

    x0 = []
    for i in range(RobotNO):
    
        yr,xr = test_env.yr_vec[i] , test_env.xr_vec[i]
        x0.append([yr,xr,0])    
    
    # for i in range(RobotNO):
    #     yr,xr = random_start(map_matrix)
    #     x0.append([yr,xr,0])
    x = np.array(x0)

    localmaps = np.multiply(0.5,np.ones((RobotNO,map_height,map_width)))
    m = np.multiply(0.5,np.ones((map_height,map_width)))
    for i in range(RobotNO):
        localmaps[i] = map_update(x[i],map_matrix,localmaps[i])
    known_points_mask = localmaps != 0.5
    for i in range(RobotNO):
        m[known_points_mask[i]] = localmaps[i][known_points_mask[i]]
    pher_map = np.ones((map_height,map_width))*0.001
    
    path_x = [[] for _ in range(RobotNO)]
    path_y = [[] for _ in range(RobotNO)]
    generated_path = [[] for _ in range(RobotNO)]
    m_for_animation = [[] for _ in range(RobotNO)]
    iter = 0 
    similarity = sim_map(map_matrix,m)
    episode_reward = 0 
    ig = 0 
    pv = 0 
    done = False
    plt.imshow(np.subtract(1,m),cmap='gray',vmin=0,vmax=1,origin='lower')
    for inde in range(RobotNO):
        plt.plot(x[inde][1],x[inde][0],'o')
    plt.show()

   

    while done == False:
        # print("Summation of pheromone map",np.sum(pher_map))    
        similarity = sim_map(map_matrix,m)
        iter += 1
        if RobotNO == 1:
            x_frontier ,  y_frontier = frontier_space((m), number_of_clusters , x[0])
        else:   
            x_frontier , y_frontier = frontier_space_nonlearning(m,number_of_clusters,x[0])
        # print("old",x_frontier)

        #plotting the frontier points
        for i in range(RobotNO):
            plt.imshow(np.subtract(1,m),cmap='gray',vmin=0,vmax=1,origin='lower')
            plt.plot(x[0][1],x[0][0],'o',color='red')
            plt.plot(x_frontier,y_frontier,'o',color='blue')
        plt.show()


        if hybrid == True and similarity < 0.6:
            for robot_id in range(RobotNO):
                for _ in range(2):
                    # print("robot_id",robot_id,"iter",_)
                    graph = RRTGraph((int(x[robot_id][0]),int(x[robot_id][1])),localmaps[robot_id])
                    while not graph.path_to_goal():
                        X, Y, Parent = graph.expand()
                    final_path = graph.getPathCoords()
                    final_path.reverse()
                    # print("final path",final_path)
                    x_path = [coord[0] for coord in final_path]
                    y_path = [coord[1] for coord in final_path]

                    if map_matrix[int(y_path[-1])][int(x_path[-1])] < 0.5:
                        x_frontier = np.append(x_frontier,x_path[-1])
                        y_frontier = np.append(y_frontier,y_path[-1])

               

        normi,normt,normcheck = bids_for_task_allocator(x_frontier,y_frontier,RobotNO,x,m,localmaps)
        normpher = calc_normpher_nonlearning(x,x_frontier,y_frontier,RobotNO,pher_map)
        frontier_index = task_allocator(x_frontier,y_frontier,RobotNO, normi,normt,normcheck,normpher,alpha ,betta,gamma,lamb)

        print("frontier_index",frontier_index)
        plt.imshow(np.subtract(1,m),cmap='gray',vmin=0,vmax=1,origin='lower')

        plt.plot(x_frontier,y_frontier,'o',color='blue')

        for robot_id in range(RobotNO):
            
            plt.plot(x[robot_id][1],x[robot_id][0],'o',color='orange')
            plt.plot(x_frontier[frontier_index[robot_id]],y_frontier[frontier_index[robot_id]],'o',color='green')
    
            

        plt.show()
        
        for robot_id in range(RobotNO):
            dumy_var = [int(x[robot_id][0]),int(x[robot_id][1]),int(x[robot_id][2])]
            best_frontier = [y_frontier[frontier_index[robot_id]],x_frontier[frontier_index[robot_id]]]
            try : 
                generated_path[robot_id] = generate_exploration_path(dumy_var, best_frontier, m)
            except :
                print("robot_id",robot_id,"frontier_index",frontier_index[robot_id])
                print("robot position",map_matrix[int(x[robot_id][0])][int(x[robot_id][1])])
                plt.imshow(np.subtract(1,m),cmap='gray',vmin=0,vmax=1,origin='lower')
                plt.plot(x[robot_id][1],x[robot_id][0],'o',color='red')
                plt.plot(x_frontier,y_frontier,'o',color='blue')
                plt.plot(x_frontier[frontier_index[robot_id]],y_frontier[frontier_index[robot_id]],'o',color='green')
                plt.show()
                frontier_index = np.random.randint(0,len(x_frontier)-1)
                best_frontier = [y_frontier[frontier_index[robot_id]],x_frontier[frontier_index[robot_id]]]
                generated_path[robot_id] = generate_exploration_path(dumy_var, best_frontier, map_matrix)
        
            gp = generated_path[robot_id]
            old_map = np.copy(m)
            stepx = []
            stepy = []

            x_prev = np.copy(x)

            for j in range(0,int(len(gp))):
                x[robot_id] = np.array([gp[j][0], gp[j][1], 0])
                path_x[robot_id].append(x[robot_id][1])
                path_y[robot_id].append(x[robot_id][0])
                m_for_animation[robot_id].append(np.copy(m))
                stepx = np.append(stepx,x[0][1])
                stepy = np.append(stepy,x[0][0])
                localmaps[robot_id] = map_update(x[robot_id],map_matrix,localmaps[robot_id])
                if map_matrix[int(gp[j][0])][int(gp[j][1])] >= 0.5:
                    print("Obstacle")
                    x[robot_id] = np.array([gp[j-1][robot_id], gp[j-1][robot_id], 0])
                    break
                # m = map_update(x[0],map_matrix,m)
                # known_points_mask = localmaps != 0.5
                # for i in range(RobotNO):
                #     m[known_points_mask[robot_id]] = localmaps[i][known_points_mask[robot_id]]
                #     m_for_animation[i].append(np.copy(m))

            known_points_mask = localmaps != 0.5
            for i in range(RobotNO):
                m[known_points_mask[i]] = localmaps[i][known_points_mask[i]]
                # m_for_animation[i].append(np.copy(m))
            pher_map += update_pheromone_map(pher_map,path_x[robot_id],path_y[robot_id],100,m,x[robot_id],0.8)
            
        new_map = np.copy(m)
        
        reward , ig, cost, pv = reward_space(old_map,new_map,stepx,stepy,y_frontier,x_frontier,x,x_prev,pher_map,False)

        ig += ig
        pv += cost
        episode_reward += reward

        plot_sim_fr.append(sim_map(map_matrix , m))
        plot_len_fr.append(np.max(np.array([total_len(path_x[i],path_y[i]) for i in range(RobotNO)])))

    
        if similarity >= term_criteria or iter >= 100 :
            r , totallen , sim  = test_env.finish_reward(m,path_x[0],path_y[0],iter)
            all_cost.append(totallen)
            all_sim.append(sim)
            all_count.append(iter)
            done = True
        
        if similarity >= 0.7 and similarity <= 0.9 :
            Topo_length.append(np.max(np.array([total_len(path_x[i],path_y[i]) for i in range(RobotNO)])))        
        
        
    
    total_len_mat =  np.array([total_len(path_x[i],path_y[i]) for i in range(RobotNO)])
    # print("total_len_mat2",total_len_mat2)
    
    # total_len_mat = np.array([total_len(path_x[0],path_y[0]),total_len(path_x[1],path_y[1]),total_len(path_x[2],path_y[2])])
    # print("total_len_mat",total_len_mat)


    all_rewards.append(episode_reward)
    all_ig.append(ig)
    all_pv.append(pv)
    
    total_path_len.append(np.sum(total_len_mat))
    time_froniter.append(np.max(total_len_mat))

    # plt.imshow(np.subtract(1,m),cmap='gray',vmin=0,vmax=1,origin='lower')
    # plt.plot(path_x[0],path_y[0],color='red')
    # plt.plot(path_x[0][0],path_y[0][0],'o',color='red')
    # plt.plot(path_x[1],path_y[1],color='blue')
    # plt.plot(path_x[1][0],path_y[1][0],'o',color='blue')
    # plt.plot(path_x[2],path_y[2],color='green')
    # plt.plot(path_x[2][0],path_y[2][0],'o',color='green')
    # plt.show() 

    return action_hist, total_len_mat , plot_sim_fr , plot_len_fr, all_rewards , [path_x[i] for i in range(RobotNO)] , [path_y[i] for i in range(RobotNO)] , Topo_length, pher_map , m_for_animation

def random_eval(test_env , map_matrix):

    TotalPath_length , all_sim , all_len, all_episode_reward  , action_hist , plot_sim , plot_len ,Topo_length = [], [],[],[],[],[],[],[]
    done, total_reward = False, 0
    next_state, x , m , localx , localy ,pher_map = test_env.reset()
    episode_step = 0
    plot_sim , plot_len , action_hist = [],[],[]
    while not done:
        episode_step += 1
        action = np.random.randint(0,9)
        action_hist.append(action)
        next_state, x , m , localx , localy, done, reward, plotinfo,pher_map = test_env.step(action,x,m,localx,localy,pher_map)
        if episode_step >= maxstep or done==True:
            done = True
            reward , totallen , sim  = test_env.finish_reward(m,localx,localy,episode_step)
        total_reward += reward

        plot_sim.append(sim_map(map_matrix,m))
        plot_len.append(total_len(localx,localy))
        
        if sim_map(map_matrix,m) >= 0.7 and sim_map(map_matrix,m) <= 0.9 :
            Topo_length.append(total_len(localx,localy))
        
        
    
    TotalPath_length.append(totallen)
    all_episode_reward.append(total_reward)
    all_sim.append(plot_sim)
    all_len.append(plot_len)
    
    return action_hist, TotalPath_length , all_sim , all_len, all_episode_reward , localx , localy , Topo_length