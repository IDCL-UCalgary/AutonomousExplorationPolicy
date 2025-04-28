import numpy as np
from environments.config import map_height, map_width, num_obstacles, obstacle_size
class Map_Generator:
    def __init__(self, seed, width = map_width, height = map_height , numb_of_obstacles = num_obstacles, obs_size = obstacle_size, margins=(2, 2)):
        self.width = width
        self.height = height
        self.numb_of_obstacles = numb_of_obstacles
        self.seed = seed
        self.obstacle_size = obs_size
        self.margins = margins
        
        np.random.seed(self.seed)
        
        self.map = np.zeros((self.width, self.height))
        self.generate_obstacles()
        self.apply_obstacles_to_map()

    def generate_obstacles(self):
        marginW, marginH = self.margins
        wMin, wMax = marginW, self.width - marginW
        hMin, hMax = marginH, self.height - marginH
        
        self.obstaclePositions = np.array([
            (np.random.randint(wMin, wMax), np.random.randint(hMin, hMax))
            for _ in range(self.numb_of_obstacles)
        ])

    def apply_obstacles_to_map(self):
        for (w, h) in self.obstaclePositions:
            obWidth = np.random.randint(self.obstacle_size[0], self.obstacle_size[0] + 1)
            obHeight = np.random.randint(self.obstacle_size[1], self.obstacle_size[1] + 1)
            
            for i in range(obWidth):
                for j in range(obHeight):
                    wI, hJ = min(w + i, self.width - 1), min(h + j, self.height - 1)
                    self.map[wI, hJ] = 1

    def ref_map(self):
        #change borders to obstacles 
        finalMap = self.map.copy()
        finalMap[0, :] = 1
        finalMap[-1, :] = 1
        finalMap[:, 0] = 1
        finalMap[:, -1] = 1
        return finalMap

