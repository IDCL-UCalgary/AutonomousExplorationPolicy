import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (LSTM, BatchNormalization, Conv2D, Dense,
                                     Dropout, Flatten, MaxPooling2D,
                                     TimeDistributed, Input,
                                     Dense, Add)
from tqdm import tqdm
from config import map_height, map_width, action_size
import numpy as np
from collections import deque
import random
from environment import env
from map_generator import Map_Generator as Generator
import matplotlib.pyplot as plt
import pandas as pd
from config import (map_height, map_width, action_size,
                    maxep,maxstep,gamma,Trainig_Episodes_NO,
                    D3QN_lr,D3QN_batch_size,D3QN_eps,D3QN_eps_decay,D3QN_eps_min,
                    D3QN_buffer_size)
tf.keras.backend.set_floatx('float64')
import warnings
import logging
warnings.filterwarnings("ignore")
tf.get_logger().setLevel(logging.ERROR)

CUR_EPISODE, all_rewards, all_cost, all_sim, all_count, all_ig, all_step_cost, all_pv, eps_value, action_mean , action_std , action_max , action_min  = 0, [], [], [],[], [], [], [], [] , [] , [], [],[]






class ReplayBuffer:
    def __init__(self, capacity=D3QN_buffer_size):
        self.buffer = deque(maxlen=capacity)
    
    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])
    
    def sample(self):
        sample = random.sample(self.buffer, D3QN_batch_size)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample)) # zip(*sample) = ([s1, s2, ...], [a1, a2, ...], ...) #
        states = np.array(states).reshape(D3QN_batch_size,map_width , map_height , 1) # (batch_size, state_dim) #
        next_states = np.array(next_states).reshape(D3QN_batch_size,map_width , map_height , 1) 
        return states, actions, rewards, next_states, done
    
    def size(self):
        return len(self.buffer)

class ActionStateModel:
    def __init__(self, state_dim, aciton_dim):
        self.state_dim  = state_dim
        self.action_dim = aciton_dim
        self.epsilon = D3QN_eps
        
        self.model = self.create_model()
    
    def create_cnn(self):
        return tf.keras.Sequential([
            Input((self.state_dim[0] , self.state_dim[1] , self.state_dim[2])),
            (Conv2D(32, kernel_size=(8,8), strides=(4,4), activation='relu')),
            (Conv2D(64 , kernel_size=(4,4), strides=(2,2), activation='relu')),
            (Conv2D(128 , kernel_size=(4,4), strides=(2,2), activation='relu')),
            (Flatten()),
        ])
    
    def create_model(self):
        cnn = self.create_cnn()
        cnn.build((None, self.state_dim[0] , self.state_dim[1] , self.state_dim[2]))
        y = Dense(512, activation='relu')(cnn.output)
        value_output = Dense(1)(y)
        advantage_output = Dense(self.action_dim)(y)
        output = Add()([value_output, advantage_output])
        model = tf.keras.Model(inputs=[cnn.input], outputs=output)
        model.compile(loss='mse', optimizer=Adam(D3QN_lr))
        return model

    
    
    
    def predict(self, state):
        return self.model.predict(state, verbose=None)
    
    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim[0] , self.state_dim[1] , self.state_dim[2]])
        self.epsilon *= D3QN_eps_decay
        self.epsilon = max(self.epsilon, D3QN_eps_min)
        q_value = self.predict(state)[0]
        if np.random.random() < self.epsilon:
            return random.randint(0, self.action_dim-1) , self.epsilon
        return np.argmax(q_value) , self.epsilon
    
    def get_action_test(self, state):
        state = np.reshape(state, [1, self.state_dim[0] , self.state_dim[1] , self.state_dim[2]])
        q_value = self.predict(state)[0]
        return np.argmax(q_value)

    def train(self, states, targets):
        self.model.fit(states, targets, epochs=1, verbose=None)
    

class Agent:
    def __init__(self,action_setting,pher_condition):
        self.env_number = action_setting
        self.pher_condtion = pher_condition
        map_generator = Generator(np.random.randint(0,100000))
        self.env = env(map_generator.ref_map(),self.env_number,self.pher_condtion)
        self.state_dim = self.env.state_dim
        self.action_dim = self.env.action_dim
        

        self.model = ActionStateModel(self.state_dim, self.action_dim)
        self.target_model = ActionStateModel(self.state_dim, self.action_dim)
        self.target_update()

        self.buffer = ReplayBuffer()
        self.max_steps = maxstep

    def target_update(self):
        weights = self.model.model.get_weights()
        self.target_model.model.set_weights(weights)
    
    def replay(self):
        for _ in range(10):
            states, actions, rewards, next_states, done = self.buffer.sample()
            targets = self.target_model.predict(states)
            next_q_values = self.target_model.predict(next_states)[range(D3QN_batch_size),np.argmax(self.model.predict(next_states), axis=1)]
            targets[range(D3QN_batch_size), actions] = rewards + (1-done) * next_q_values * gamma
            self.model.train(states, targets)
            # print("Training is done!")
    
    def train(self, max_episodes=Trainig_Episodes_NO):

        global CUR_EPISODE, all_rewards, all_cost, all_sim, all_count, all_ig, all_step_cost, all_pv, eps_value,action_mean , action_std , action_max , action_min

        t = tqdm(range(max_episodes))
        for ep in t:
            self.env = env(Generator(np.random.randint(0,100000)).ref_map(),self.env_number,self.pher_condtion)
            episode_reward, done = 0, False
            state , x , m , localx , localy, pher_map = self.env.reset()

            reward = 0 
            action = 0 
            episode_step = 0
            ig = 0
            step_cost =0
            pv = 0 
            action_hist = [] 

            while not done:
                episode_step += 1 
                action , epsilon_value = self.model.get_action(state)
                action_hist.append(action)
                next_state, x , m , localx , localy , done ,reward , plotinfo, pher_map = self.env.step(action,x,m,localx,localy, pher_map)
                self.buffer.put(state, action, reward, next_state, done)
                if episode_step >= self.max_steps or done==True:
                    done = True
                    r , totallen , sim  = self.env.finish_reward(m,localx,localy,episode_step)
                    reward += r 

                episode_reward += reward
                state = next_state

                ig += plotinfo[0]
                step_cost += plotinfo[1]
                pv += plotinfo[2]
                
            
            
            eps_value.append(epsilon_value)
            all_rewards.append(episode_reward)
            all_ig.append(ig)
            all_pv.append(pv)
            all_cost.append(totallen)
            all_sim.append(sim)
            all_count.append(episode_step)
            action_mean.append(np.mean(action_hist))
            action_std.append(np.std(action_hist))
            action_max.append(np.max(action_hist))
            action_min.append(np.min(action_hist))
            all_step_cost.append(step_cost)


            if self.buffer.size() >= D3QN_batch_size:
                self.replay()                
            self.target_update()
            if ep % 100 == 0:
        
                dict = {'Episode Reward': all_rewards,'Information Gain':all_ig,'Pheromone value':all_pv,'Step Cost':all_step_cost,'Cost':all_cost,
                    'Similarity':all_sim,'Count':all_count, 'Action Mean':action_mean,'Action Std':action_std,
                    'Action Max':action_max, 'Action Min':action_min}
                df = pd.DataFrame(dict) 
                df.to_csv('D3QN_Info.csv')
                self.model.model.save_weights('./D3QN/my_checkpoint_Actor')

                np.save('./maps/D3QN_map_{}.npy'.format(CUR_EPISODE),m)
                np.save('./maps/D3QN_localx_{}.npy'.format(CUR_EPISODE),localx)
                np.save('./maps/D3QN_localy_{}.npy'.format(CUR_EPISODE),localy)

            
def eval_agent(test_env,map_matrix):
    from utils import sim_map,total_len
    TotalPath_length , all_sim , all_len, all_episode_reward  , action_hist , plot_sim , plot_len , Topo_length = [],[],[],[],[],[],[],[]
    episode_reward, done = 0, False
    state , x , m , localx , localy,pher_map  = test_env.reset()
    reward = 0 
    action = 0 
    episode_step = 0
    
    
    agent = Agent(1,False)
    agent.model.model.load_weights('./D3QN/my_checkpoint_Actor')
    while not done:    
        episode_step += 1 
        action = agent.model.get_action_test(state)
        action_hist.append(action)
        next_state, x , m , localx , localy , done ,reward , plotinfo, pher_map = test_env.step(action,x,m,localx,localy, pher_map)
        if episode_step >= agent.max_steps or done==True:
            done = True
            r , totallen , sim  = test_env.finish_reward(m,localx,localy,episode_step)
            reward += r 

        episode_reward += reward
        state = next_state
        plot_sim.append(sim_map(map_matrix,m))
        plot_len.append(total_len(localx,localy))
        if sim_map(map_matrix,m) >= 0.7 and sim_map(map_matrix,m) <= 0.9 :
            Topo_length.append(total_len(localx,localy))


    TotalPath_length.append(totallen)
    all_sim.append(plot_sim)
    all_len.append(plot_len)
    all_episode_reward.append(episode_reward)

    return action_hist, TotalPath_length , all_sim , all_len, all_episode_reward , localx , localy, Topo_length

    


def main():
    env_number = 1 
    pher_condition = False
    agent = Agent(env_number,pher_condition)
    agent.train(max_episodes=Trainig_Episodes_NO)

if __name__ == "__main__":
    main()