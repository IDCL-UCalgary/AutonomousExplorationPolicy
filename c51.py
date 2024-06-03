import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Softmax
from tensorflow.keras.optimizers import Adam

import numpy as np
from collections import deque
import random
import math
import pandas as pd 

from config import (map_height, map_width, action_size,
                    maxep,maxstep,gamma,Trainig_Episodes_NO,
                    D3QN_lr,D3QN_batch_size,D3QN_eps,D3QN_eps_decay,D3QN_eps_min,
                    D3QN_buffer_size)

from tensorflow.keras.layers import (LSTM, BatchNormalization, Conv2D, Dense,
                                     Dropout, Flatten, MaxPooling2D,
                                     TimeDistributed, Input,
                                     Dense, Add)

from environment import env
from map_generator import Map_Generator as Generator
from tqdm import tqdm
tf.keras.backend.set_floatx('float64')
import warnings
import logging
warnings.filterwarnings("ignore")
tf.get_logger().setLevel(logging.ERROR)
CUR_EPISODE, all_rewards, all_cost, all_sim, all_count, all_ig, all_step_cost, all_pv, eps_value, action_mean , action_std , action_max , action_min  = 0, [], [], [],[], [], [], [], [] , [] , [], [],[]


gamma = 0.99
lr = 0.0001
batch_size = 6
atoms = 51
v_min = -4.0
v_max = 4.0





class ReplayBuffer:
    def __init__(self, capacity=4000):
        self.buffer = deque(maxlen=capacity)

    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self):
        sample = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, done = map(
            np.asarray, zip(*sample))
        states = np.array(states).reshape(batch_size,map_width , map_height , 1) # (batch_size, state_dim) #
        next_states = np.array(next_states).reshape(batch_size,map_width , map_height , 1) 
        return states, actions, rewards, next_states, done

    def size(self):
        return len(self.buffer)


class ActionValueModel:
    def __init__(self, state_dim, action_dim, z):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.atoms = atoms
        self.z = z

        self.opt = tf.keras.optimizers.Adam(lr)
        self.criterion = tf.keras.losses.CategoricalCrossentropy()
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
        h2 = Dense(64, activation='relu')(y)
        outputs = []
        for _ in range(self.action_dim):
            outputs.append(Dense(self.atoms, activation='softmax')(h2))
        model = tf.keras.Model(cnn.input, outputs)
        return model

    def train(self, x, y):
        y = tf.stop_gradient(y)
        with tf.GradientTape() as tape:
            logits = self.model(x)
            loss = self.criterion(y, logits)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

    def predict(self, state):
        return self.model.predict(state,verbose=None)

    def get_action(self, state, ep):
        state = np.reshape(state, [1, self.state_dim[0] , self.state_dim[1] , self.state_dim[2]])
        eps = 1. / ((ep / 10) + 1)
        if np.random.rand() < eps:
            return np.random.randint(0, self.action_dim)
        else:
            return self.get_optimal_action(state)
    
    def get_action_test(self, state):
        state = np.reshape(state, [1, self.state_dim[0] , self.state_dim[1] , self.state_dim[2]])
        return self.get_optimal_action(state)

    def get_optimal_action(self, state):
        z = self.model.predict(state,verbose=None)
        z_concat = np.vstack(z)
        q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1)
        return np.argmax(q)


class Agent:
    def __init__(self, action_setting, pher_condition):
        self.env_number = action_setting
        self.pher_condtion = pher_condition
        map_generator = Generator(np.random.randint(0,100000))
        self.env = env(map_generator.ref_map(),self.env_number,self.pher_condtion)
        self.state_dim = self.env.state_dim
        self.action_dim = self.env.action_dim
        self.buffer = ReplayBuffer()
        self.batch_size = batch_size
        self.v_max = v_max
        self.v_min = v_min
        self.atoms = atoms
        self.delta_z = float(self.v_max - self.v_min) / (self.atoms - 1)
        self.z = [self.v_min + i * self.delta_z for i in range(self.atoms)]
        self.gamma = gamma
        self.q = ActionValueModel(self.state_dim, self.action_dim, self.z)
        self.q_target = ActionValueModel(
            self.state_dim, self.action_dim, self.z)
        self.target_update()
        self.max_steps = maxstep

    def target_update(self):
        weights = self.q.model.get_weights()
        self.q_target.model.set_weights(weights)

    def replay(self):
        states, actions, rewards, next_states, dones = self.buffer.sample()
        # print(states.shape)
        # print(next_states.shape)
        z = self.q.predict(next_states)
        z_ = self.q_target.predict(next_states)
        z_concat = np.vstack(z)
        q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1)
        q = q.reshape((self.batch_size, self.action_dim), order='F')
        next_actions = np.argmax(q, axis=1)
        m_prob = [np.zeros((self.batch_size, self.atoms))
                  for _ in range(self.action_dim)]
        for i in range(self.batch_size):
            if dones[i]:
                Tz = min(self.v_max, max(self.v_min, rewards[i]))
                bj = (Tz - self.v_min) / self.delta_z
                l, u = math.floor(bj), math.ceil(bj)
                m_prob[actions[i]][i][int(l)] += (u - bj)
                m_prob[actions[i]][i][int(u)] += (bj - l)
            else:
                for j in range(self.atoms):
                    Tz = min(self.v_max, max(
                        self.v_min, rewards[i] + self.gamma * self.z[j]))
                    bj = (Tz - self.v_min) / self.delta_z
                    l, u = math.floor(bj), math.ceil(bj)
                    m_prob[actions[i]][i][int(
                        l)] += z_[next_actions[i]][i][j] * (u - bj)
                    m_prob[actions[i]][i][int(
                        u)] += z_[next_actions[i]][i][j] * (bj - l)
        self.q.train(states, m_prob)

    def train(self, max_episodes=Trainig_Episodes_NO):
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
                action = self.q.get_action(state, ep)
                action_hist.append(action)
                next_state, x , m , localx , localy , done ,reward , plotinfo, pher_map = self.env.step(action,x,m,localx,localy, pher_map)
                if episode_step >= self.max_steps or done==True:
                    done = True
                    r , totallen , sim  = self.env.finish_reward(m,localx,localy,episode_step)
                    reward += r 
                self.buffer.put(state, action, reward, next_state, done)
                
                episode_reward += reward
                state = next_state

                ig += plotinfo[0]
                step_cost += plotinfo[1]
                pv += plotinfo[2]
                # self.replay()
                if self.buffer.size() > 6:
                    self.replay()
                #if episode_step % 5 == 0:
                self.target_update()
                episode_step += 1
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
            
            if ep % 100 == 0:
    
                dict = {'Episode Reward': all_rewards,'Information Gain':all_ig,'Pheromone value':all_pv,'Step Cost':all_step_cost,'Cost':all_cost,
                    'Similarity':all_sim,'Count':all_count, 'Action Mean':action_mean,'Action Std':action_std,
                    'Action Max':action_max, 'Action Min':action_min}
                df = pd.DataFrame(dict) 
                df.to_csv('C51_Info.csv')
                self.q.model.save_weights('./C51/my_checkpoint_Actor')

                np.save('./maps/C51_map_{}.npy'.format(CUR_EPISODE),m)
                np.save('./maps/C51_localx_{}.npy'.format(CUR_EPISODE),localx)
                np.save('./maps/C51_localy_{}.npy'.format(CUR_EPISODE),localy)

            

def eval_agent(test_env,map_matrix):
    from utils import sim_map,total_len
    TotalPath_length , all_sim , all_len, all_episode_reward  , action_hist , plot_sim , plot_len,Topo_length = [],[],[],[],[],[],[],[]
    episode_reward, done = 0, False
    state , x , m , localx , localy,pher_map  = test_env.reset()
    reward = 0 
    action = 0 
    episode_step = 0
    
    
    agent = Agent(1,False)
    agent.q.model.load_weights('./C51/my_checkpoint_Actor')

    while not done:    
        episode_step += 1 
        action = agent.q.get_action_test(state)
        action_hist.append(action)
        next_state, x , m , localx , localy , done ,reward , plotinfo, pher_map = test_env.step(action,x,m,localx,localy, pher_map)
        if episode_step >= agent.max_steps or done==True:
            done = True
            r , totallen , sim  = test_env.finish_reward(m,localx,localy,episode_step)
            reward += r 

        state = next_state
        episode_reward += reward
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