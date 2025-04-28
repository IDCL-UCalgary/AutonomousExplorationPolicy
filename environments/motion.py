import numpy as np
from queue import PriorityQueue
from typing import List, Tuple, Union, Optional

from environments.config import map_height, map_width
from environments.utils.geometry_utils import move, check_free_or_occupied

class MyGlobalplanner :
	def __init__(self,start_point , goal_point , method='a'):
		self.start_point = start_point
		self.goal_point = goal_point
		self.method = method
		self.generated_points = []
		self.parent = np.full(fill_value= -1, shape=(map_height,map_width))
		self.base_cost = np.full(fill_value=-1, shape=(map_height, map_width))


	def h_score(self , point):
		return np.sqrt((self.goal_point[0]-point[0])**2 + (self.goal_point[1]-point[1])**2)
	def total_score(self,point,point_cost):
		if self.method == 'a':
				return self.h_score(point) + point_cost
		elif self.method == 'd' :
				return point_cost
		elif self.method == 'b' :
				return 1
		else :
				quit()

	def find_global_path(self , map_img):
		point_queue = PriorityQueue()

		self.base_cost[self.start_point[0]][self.start_point[1]] = 0
		self.generated_points.append(self.start_point)
		self.parent[self.start_point[0]][self.start_point[1]] = -99
		point_queue.put((self.total_score(self.start_point,0),self.start_point))

		while not point_queue.empty():
			current_point = point_queue.get()
			if current_point[1] == self.goal_point:
				break
			for i in range(8):
				y,x = move(i , current_point[1])
				if (check_free_or_occupied(map_img,x,y) and self.parent[y][x] == -1 ):
					#print("entered first if!")
					if i < 4 :
						self.base_cost[y][x] = self.base_cost[current_point[1][0],current_point[1][1]] + 1
					elif i>=4 and i<8:
						self.base_cost[y][x] = self.base_cost[current_point[1][0], current_point[1][1]] + np.sqrt(2)
					elif i>= 8 and i < 12:
						self.base_cost[y][x] = self.base_cost[current_point[1][0],current_point[1][1]] + 2
					elif i < 16 : 
						self.base_cost[y][x] = self.base_cost[current_point[1][0], current_point[1][1]] + 2*np.sqrt(2)


					point_queue.put((self.total_score((y,x),self.base_cost[y][x]),(y,x)))
					self.generated_points.append((y,x))
					self.parent[y][x] = np.ravel_multi_index([current_point[1][0],current_point[1][1]],dims=(map_height,map_width))


	def generate_path(self):
		points_in_path = []
		last_point = self.goal_point
		points_in_path.append(last_point)
		while self.parent[last_point[0]][last_point[1]] != -99:
				last_point = np.unravel_index(self.parent[last_point[0]][last_point[1]],shape=(map_height,map_width))
				points_in_path.append(last_point)
		return points_in_path



def generate_exploration_path(x,frontier,m):
	goal_point_coords = (x[0],x[1])
	start_point_coords = (frontier[0],frontier[1])
	Global_Path = MyGlobalplanner(start_point_coords,goal_point_coords,str('a'))
	Global_Path.find_global_path(m)
	generated_path = np.array(Global_Path.generate_path())
	return generated_path


def generate_exploration_path_rrt(x,x_goal,y_goal,m):
    goal_point_coords = (x[0],x[1])
    start_point_coords = (y_goal,x_goal)
    Global_Path = MyGlobalplanner(start_point_coords,goal_point_coords,str('a'))
    #Global_Path.find_global_path(obstacle_inflation(map_matrix))
    Global_Path.find_global_path(m) # this is new just for test
    generated_path = np.array(Global_Path.generate_path())
    return generated_path


def total_len(localx,localy):
    dist = 0
    for i in range(len(localx)-1):
        dist += np.sqrt((localx[i]-localx[i+1])**2 + (localy[i]-localy[i+1])**2)
    return dist
