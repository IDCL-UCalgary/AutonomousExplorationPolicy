import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from config import action_size, map_height, map_width , normigfactor , rmax, meas_phi , number_of_clusters
from mapping import map_update , get_range
from motion import generate_exploration_path
from utils import (calc_heading_error, calc_pher, finishreward,
                   frontier_detection, frontier_space, obstacle_inflation,
                   obstacle_inflation_narrow, random_start, repeat_cluster,
                   reward_space, select_nearest_frontier, sig_point, sim_map,
                   total_len, update_pheromone_map,count_unkonwn_cells,action_to_goal_case3,calc_measurements , find_nearest_free_space)
from time import time
# import matplotlib.pyplot as plt
class env:
    def __init__(self,map_matrix,action_setting,pher_condition,number_of_robots=1):
        self.env_number = action_setting
        self.map_matrix = map_matrix
        self.map_height = map_height
        self.map_width = map_width
        self.pher_condition = pher_condition
        self.action_dim = action_size
        self.num_of_clusters = number_of_clusters
        self.number_of_robots = number_of_robots
        self.yr_vec , self.xr_vec = [] , []
        for _ in range(number_of_robots):
            yr , xr = random_start(self.map_matrix)
            self.yr_vec.append(yr)
            self.xr_vec.append(xr)
        # self.xr_vec  = [random_start(self.map_matrix)[1] for i in range(number_of_robots)]
        if self.number_of_robots == 1:
            self.yr , self.xr = self.yr_vec[0] , self.xr_vec[0]

        
        if self.pher_condition == True:
            self.state_dim = (map_height,map_width,2)
        else : 
            self.state_dim = (map_height,map_width,1)
        

    def reset(self):
        
        yr,xr = self.yr , self.xr
        x0 = [yr,xr,0]
        x = np.array([x0])
        RobotNO = len(x)
        m = np.multiply(0.5,np.ones((self.map_height,self.map_width)))
        for i in range(RobotNO):
            m = map_update(x[i],self.map_matrix,m)
        pher_map = np.zeros((map_height,map_width))
        localx = []
        localy = []
        x_frontiers ,  y_frontiers = frontier_space(m, self.num_of_clusters , x[0])
        if self.pher_condition == True:
            state = np.zeros((map_height,map_width,2))
            state[:,:,0] = np.copy(m)
            state[x[0][0],x[0][1],0] = 0.8
            for i in range(len(x_frontiers)):
                state[y_frontiers[i],x_frontiers[i],0] = 0.3
            state[:,:,1] = np.copy(pher_map)
        elif self.pher_condition == False: 
            state = np.copy(m)
            state[x[0][0],x[0][1]] = 0.8
            for i in range(len(x_frontiers)):
                state[y_frontiers[i]][x_frontiers[i]] = 0.3
            
        
        return state , x , m , localx , localy , pher_map
    

    def step(self,action,x,m,localx,localy,pher_map):
        done = False
        r = 0 # this one is used in env 4 but for all other envs its zero
        if self.env_number == 1 :
            x_frontiers ,  y_frontiers = frontier_space((m), self.num_of_clusters , x[0])
            distances = np.sqrt((x[0][0]-y_frontiers)**2 +(x[0][1]-x_frontiers)**2)
            sorted_indices = np.argsort(distances)
            goal_idx = sorted_indices[action]
            goal = [int(y_frontiers[goal_idx]), int(x_frontiers[goal_idx]), 0]
        elif self.env_number == 2 :
            x_frontiers ,  y_frontiers = frontier_space((m), self.num_of_clusters , x[0])
            goal_idx = action
            goal = [int(y_frontiers[goal_idx]), int(x_frontiers[goal_idx]), 0]
        elif self.env_number == 3 :
            x_frontiers ,  y_frontiers= [] , []
            for ii in range(action_size):
                yyy, xxx , _ , _= action_to_goal_case3((ii*360/action_size),obstacle_inflation_narrow(m),x[0])
                x_frontiers.append(xxx)
                y_frontiers.append(yyy)
            y_g , x_g , _ , _  = action_to_goal_case3((action*360/(action_size)),obstacle_inflation(m),x[0])
            goal = [int(y_g), int(x_g), 0]
        elif self.env_number == 4 :
            x_frontiers , y_frontiers = [] , []
            for ii in range(action_size):
                yyy, xxx , r , done= action_to_goal_case3((ii*360/action_size),obstacle_inflation_narrow(m),x[0])
                x_frontiers.append(xxx)
                y_frontiers.append(yyy)
            y_g , x_g , r , done  = action_to_goal_case3((action*360/(action_size)),obstacle_inflation(m),x[0])
            goal = [int(y_g), int(x_g), 0]
             
        dumy_var = [int(x[0][0]),int(x[0][1]),int(x[0][2])]

        try : 
            generated_path = generate_exploration_path(dumy_var, goal, (m))
        except :
            try :
                # print("No path 1")
                generated_path = generate_exploration_path(dumy_var, goal, self.map_matrix)
            except:
                # print("No path 2")
                generated_path = []
                done = True
        
        old_map = np.copy(m)
        stepx = []
        stepy = []

        x_prev = np.copy(x)

    
        for j in range(0,int(len(generated_path))):
            x[0] = np.array([generated_path[j][0], generated_path[j][1], 0])
            localx = np.append(localx,x[0][1])
            localy = np.append(localy,x[0][0])
            stepx = np.append(stepx,x[0][1])
            stepy = np.append(stepy,x[0][0])
            m = map_update(x[0],self.map_matrix,m)
            if self.map_matrix[int(generated_path[j][0])][int(generated_path[j][1])] >= 0.5:
                # print("Obstacle")
                x[0] = np.array([generated_path[j-1][0], generated_path[j-1][1], 0])
                break
        
        if len(stepx) == 0  :
            stepx = np.append(stepx,x[0][1])
            stepy = np.append(stepy,x[0][0])
            m = map_update(x[0],self.map_matrix,m)
        
        new_map = np.copy(m)
        reward , ig, cost, pv = reward_space(old_map,new_map,stepx,stepy,y_frontiers,x_frontiers,x,x_prev,pher_map,self.pher_condition)
        reward += r
        pher_map = update_pheromone_map(pher_map,stepx,stepy,100,m,x[0],decay_rate=0.8)


        if sim_map(self.map_matrix, m) > 0.95 :
            done = True
       

        if self.pher_condition == True :

            state = np.zeros((map_height,map_width,2))
            state[:,:,0] = np.copy(m)
            state[x[0][0],x[0][1],0] = 0.8

            if self.env_number == 1  or self.env_number == 2 :
                for i in range(len(x_frontiers)):
                    state[y_frontiers[i],x_frontiers[i],0] = 0.3
                state[:,:,1] = np.copy(pher_map)
            
            elif self.env_number == 3 or self.env_number == 4:
                for i in range(action_size):
                    yyy, xxx, _ , _ = action_to_goal_case3((i*360/action_size),(m),x[0])
                    state[int(yyy),int(xxx),0] = 0.3
                state[:,:,1] = np.copy(pher_map)

        
        elif self.pher_condition == False:
            state = np.copy(m)
            state[x[0][0],x[0][1]] = 0.8
            if self.env_number == 1  or self.env_number == 2 :
                x_frontiers ,  y_frontiers = frontier_space(m, self.num_of_clusters , x[0])
                for i in range(len(x_frontiers)):
                    state[y_frontiers[i]][x_frontiers[i]] = 0.3
            elif self.env_number == 3 or self.env_number == 4:
                for i in range(action_size):
                    yyy, xxx, _ , _ = action_to_goal_case3((i*360/action_size),(m),x[0])
                    state[int(yyy),int(xxx)] = 0.3
        return state, x , m  , localx , localy , done ,reward , np.array([ig,cost,pv]) , pher_map
    

    def finish_reward(self,m,localx,localy,episode_step):
        reward , totallen , sim = finishreward(m,localx,localy,self.map_matrix)

        return reward , totallen , sim  



    # def bids_for_task_allocator(self , x_frontier,y_frontier,number_of_robots,x,m):
    #     T = np.zeros([number_of_robots,len(x_frontier)])
    #     I = np.zeros([number_of_robots,len(x_frontier)])
    #     check = np.ones([number_of_robots,len(x_frontier)])*0.001
    #     pheromnelist = np.zeros(len(x_frontier))

    #     for i in range(number_of_robots):
    #         T[i] = np.sqrt((x[i][0]-y_frontier)**2 +(x[i][1]-x_frontier)**2)
            
    #     for j in range(len(x_frontier)):
    #         count = 0 
    #         for xx in range(int(x_frontier[j])-(rmax+1),int(x_frontier[j])+(rmax+2)):
    #                 for yy in range(int(y_frontier[j])-(rmax+1),int(y_frontier[j])+(rmax+2)):
    #                         if xx<0 or xx> map_width-1 or yy<0 or yy>map_height-1 :
    #                                 pass 
    #                         else :
    #                                 if ((xx-x_frontier[j])**2 + (yy-y_frontier[j])**2)<= rmax**2 : 
    #                                         if m[yy][xx] == 0.5:
    #                                                 count += 1
    #         I[0][j] = count
                
    #     normi = I/np.linalg.norm(I)
    #     normt = T/np.linalg.norm(T)
    #     normcheck = np.zeros([number_of_robots,len(x_frontier)])
    #     return normi,normt,normcheck
    
    
    
    # def task_allocator(self, x_frontier,y_frontier,RobotNO,normi,normt,normcheck,normpher,alpha=0.0 ,betta=1.5,gamma=1,lamb = 1):
    #     frontier_index = np.zeros(RobotNO,int)
    #     for i in range(RobotNO):
    #         p = alpha*normi - betta*normt - gamma*normcheck - lamb*normpher
    #         frontier_index[i] = np.where(p[i] == np.amax(p[i]))[0][0]
    #         for j in range(RobotNO-(i+1)):
    #             for k in range(len(x_frontier)):
    #                 if ((x_frontier[frontier_index[i]]-x_frontier[k])**2 + (y_frontier[frontier_index[i]]-y_frontier[k])**2)<=(1.5*rmax)**2:
    #                     normt[j+i+1][k] *= 5
    #     return frontier_index
    
    
    # def nonlearningapproach(self , alpha = 0.8 ,betta=0.2,gamma=0,lamb = 0.0):
    #     iter_only_frontier = []
    #     total_path_len = []
    #     igval_frontier = []
    #     time_froniter = [] 
    #     term_criteria = 0.95
    #     all_rewards = []
    #     step_rewards = [] 
    #     all_ig = [] 
    #     all_pv = [] 
    #     all_cost = [] 
    #     all_sim = []
    #     all_count = []
    #     all_utility = []

    #     yr,xr = yr,xr = self.yr , self.xr
    #     x0 = [yr,xr,0]
    #     x = np.array([x0])
    #     RobotNO = len(x)

    #     m = np.multiply(0.5,np.ones((self.map_height,self.map_width)))
    #     for i in range(RobotNO):
    #         m = map_update(x[i],self.map_matrix,m)



    #     pher_map = np.ones((map_height,map_width))*0.001
        



    #     path_x = [[],[],[],[],[],[]]
    #     path_y = [[],[],[],[],[],[]]
    #     generated_path = [[],[],[],[],[],[]]
    #     iter = 0 
    #     similarity = sim_map(self.map_matrix,m)
    #     episode_reward = 0 
    #     ig = 0 
    #     pv = 0 
    #     step_rewards.append(episode_reward)
    #     done = False

    #     plot_sim_fr = []
    #     plot_len_fr = []

    #     while done == False:
    #         similarity = sim_map(self.map_matrix,m)
    #         iter += 1
    #         x_frontier ,  y_frontier = frontier_space((m), self.num_of_clusters , x[0])
    #         normi,normt,normcheck = self.bids_for_task_allocator(x_frontier,y_frontier,RobotNO,x,m)
    #         normpher = np.zeros([RobotNO,len(x_frontier)])
    #         frontier_index = self.task_allocator(x_frontier,y_frontier,RobotNO, normi,normt,normcheck,normpher,alpha ,betta,gamma,lamb)
    #         xxxx = 0 
    #         dumy_var = [int(x[xxxx][0]),int(x[xxxx][1]),int(x[xxxx][2])]
    #         best_frontier = [y_frontier[frontier_index[xxxx]],x_frontier[frontier_index[xxxx]]]
    #         try : 
    #             generated_path[xxxx] = generate_exploration_path(dumy_var, best_frontier, m)
    #         except :
    #             frontier_index = np.random.randint(0,len(x_frontier)-1)
    #             best_frontier = [y_frontier[frontier_index[xxxx]],x_frontier[frontier_index[xxxx]]]
    #             generated_path[xxxx] = generate_exploration_path(dumy_var, best_frontier, self.map_matrix)
        
    #         gp = generated_path[xxxx]
    #         old_map = np.copy(m)
    #         stepx = []
    #         stepy = []

    #         x_prev = np.copy(x)

    #         for j in range(0,int(len(gp))):
    #             x[xxxx] = np.array([gp[j][0], gp[j][1], 0])
    #             path_x[xxxx].append(x[xxxx][1])
    #             path_y[xxxx].append(x[xxxx][0])
    #             stepx = np.append(stepx,x[0][1])
    #             stepy = np.append(stepy,x[0][0])
    #             m = map_update(x[0],self.map_matrix,m)
            
            
    #         new_map = np.copy(m)
            
    #         reward , ig, cost, pv = reward_space(old_map,new_map,stepx,stepy,y_frontier,x_frontier,x,x_prev,pher_map,self.pher_condition)

    #         ig += ig
    #         pv += cost
    #         step_rewards.append(reward)
    #         episode_reward += reward

    #         plot_sim_fr.append(sim_map(self.map_matrix , m))
    #         plot_len_fr.append(total_len(path_x[0],path_y[0]))

        
    #         if similarity >= term_criteria or iter >= 200 :
    #             r , totallen , sim  = self.finish_reward(m,path_x[0],path_y[0],iter)
    #             all_cost.append(totallen)
    #             all_sim.append(sim)
    #             all_count.append(iter)
    #             # reward += r
    #             # episode_reward += r
    #             step_rewards.append(reward)
    #             done = True
            
           
            
    #     total_len_mat = np.array([total_len(path_x[0],path_y[0]),total_len(path_x[1],path_y[1]),total_len(path_x[2],path_y[2]),total_len(path_x[3],path_y[3]),total_len(path_x[4],path_y[4]),total_len(path_x[4],path_y[4])])

    #     all_rewards.append(episode_reward)
    #     all_ig.append(ig)
    #     all_pv.append(pv)
        
    #     total_path_len.append(np.sum(total_len_mat))
    #     iter_only_frontier.append(iter)
    #     time_froniter.append(np.max(total_len_mat))


    #     return plot_sim_fr , plot_len_fr , total_len_mat[0] , path_x[0] , path_y[0] , all_rewards 
