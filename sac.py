import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.optimizers import Adam
# import matplotlib.pyplot as plt
import numpy as np
import random
import tqdm

from environment import env
import warnings
warnings.filterwarnings("ignore")
from map_generator import Generator
from config import (map_width, map_height, number_of_clusters, action_size,
                    sac_lr , sac_batch_size, sac_buffer_size, sac_tau,sac_alpha,
                    gamma,Trainig_Episodes_NO,maxep,maxstep)
import pandas as pd

from tensorflow.keras.layers import (LSTM, BatchNormalization, Conv2D, Dense,
                                     Dropout, Flatten, MaxPooling2D,
                                     TimeDistributed)

tf.keras.backend.set_floatx('float32')




# Policy net (pi_theta)
class PolicyNet(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
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
        y = Dense(self.action_dim, activation='linear')(y)
        model = tf.keras.Model(inputs=[cnn.input], outputs=y)
        return model
    
    
    def call(self, inputs):
        return self.model(np.reshape(inputs, [len(inputs), self.state_dim[0] , self.state_dim[1] , self.state_dim[2]]))
    
class QNet(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(QNet, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = self.create_model()

    def create_cnn(self):
        return tf.keras.Sequential([
            Input((self.state_dim[0] , self.state_dim[1] , 1)),
            (Conv2D(32, kernel_size=(8,8), strides=(4,4), activation='relu')),
            (Conv2D(64 , kernel_size=(4,4), strides=(2,2), activation='relu')),
            (Conv2D(128 , kernel_size=(4,4), strides=(2,2), activation='relu')),
            (Flatten()),
        ])
    
    def create_model(self):
        cnn = self.create_cnn()
        cnn.build((None, self.state_dim[0] , self.state_dim[1] , 1))
        y = Dense(512, activation='relu')(cnn.output)
        y = Dense(self.action_dim, activation='linear')(y)
        model = tf.keras.Model(inputs=[cnn.input], outputs=y)
        return model
    
    
    
    def call(self, inputs):
        return self.model(np.reshape(inputs, [len(inputs), self.state_dim[0],self.state_dim[1],self.state_dim[2]]))

class Categorical: 
    def __init__(self, s):
        logits = pi_model(s)
        self._prob = tf.nn.softmax(logits)
        self._logp = tf.math.log(self._prob)

    def prob(self):
        return self._prob

    def logp(self):
        return self._logp

class ReplayBuffer:
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.buffer = []
        self._next_idx = 0

    def add(self, item):
        if len(self.buffer) > self._next_idx:
            self.buffer[self._next_idx] = item
        else:
            self.buffer.append(item)
        if self._next_idx == self.buffer_size - 1:
            self._next_idx = 0
        else:
            self._next_idx = self._next_idx + 1

    def sample(self, batch_size):
        indices = [random.randint(0, len(self.buffer) - 1) for _ in range(batch_size)]
        states   = [self.buffer[i][0] for i in indices]
        actions  = [self.buffer[i][1] for i in indices]
        rewards  = [self.buffer[i][2] for i in indices]
        n_states = [self.buffer[i][3] for i in indices]
        dones    = [self.buffer[i][4] for i in indices]
        return states, actions, rewards, n_states, dones

    def length(self):
        return len(self.buffer)


def pick_sample(s):
    s_batch = tf.expand_dims(s, axis=0)
    logits = pi_model(s_batch)
    logits = tf.reshape(logits, [1, action_dim])  # Reshape logits to [1, 2]
    probs = tf.nn.softmax(logits)
    a = tf.random.categorical(tf.math.log(probs), num_samples=1)
    a = tf.squeeze(a, axis=0)
    return a.numpy().item()


def optimize_theta(states):
    with tf.GradientTape() as tape:
        states = tf.convert_to_tensor(states)
        dist = Categorical(states)
        q_value = q_origin_model1(states)
        term1 = dist.prob()
        term2 = q_value - sac_alpha * dist.logp()
        expectation = term1[:, tf.newaxis,:] @ term2[:, :,tf.newaxis]
        expectation = tf.squeeze(expectation, axis=1)
        loss = -tf.reduce_sum(expectation)
    gradients = tape.gradient(loss, pi_model.trainable_variables)
    opt_pi.apply_gradients(zip(gradients, pi_model.trainable_variables))

def optimize_phi(states, actions, rewards, next_states, dones):
    states = tf.convert_to_tensor(states ,dtype=tf.float32)
    actions = tf.convert_to_tensor(actions,dtype=tf.int32)
    rewards = tf.convert_to_tensor(rewards,dtype=tf.float32)
    rewards = rewards[:, tf.newaxis]
    next_states = tf.convert_to_tensor(next_states,dtype=tf.float32)
    dones = tf.convert_to_tensor(dones,dtype=tf.float32)
    dones = dones[:, tf.newaxis]


    with tf.GradientTape() as tape:
        q1_tgt_next = q_target_model1(next_states)
        q2_tgt_next = q_target_model2(next_states)
        dist_next = Categorical(next_states)
        q1_target = tf.matmul(q1_tgt_next[:, tf.newaxis,:], dist_next.prob()[:,:,tf.newaxis])
        q1_target = tf.squeeze(q1_target, axis=1)
        q2_target = tf.matmul(q2_tgt_next[:, tf.newaxis,:], dist_next.prob()[:,:,tf.newaxis])
        q2_target = tf.squeeze(q2_target, axis=1)
        q_target_min = tf.minimum(q1_target, q2_target)
        h = tf.matmul(dist_next.prob()[:, tf.newaxis,:], dist_next.logp()[:,:,tf.newaxis])
        h = tf.squeeze(h, axis=1)
        h = -sac_alpha * h
        term2 = rewards + gamma * (1.0 - dones) * (q_target_min + h)

        # Optimize critic loss for Q-network1
        one_hot_actions = tf.one_hot(actions, depth=action_dim)
        q_value1 = q_origin_model1(states)
        term1 = tf.matmul(q_value1[:, tf.newaxis,:], one_hot_actions[:,:,tf.newaxis])
        term1 = tf.squeeze(term1, axis=1)
        loss_q1 = tf.losses.mean_squared_error(term2, term1)

        # Optimize critic loss for Q-network2
        q_value2 = q_origin_model2(states)
        term1 = tf.matmul(q_value2[:, tf.newaxis,:], one_hot_actions[:,:,tf.newaxis])
        term1 = tf.squeeze(term1, axis=1)
        loss_q2 = tf.losses.mean_squared_error(term2, term1)

        loss = loss_q1 + loss_q2

    gradients = tape.gradient(loss, q_origin_model1.trainable_variables + q_origin_model2.trainable_variables)
    opt_q1.apply_gradients(zip(gradients[:len(q_origin_model1.trainable_variables)], q_origin_model1.trainable_variables))
    opt_q2.apply_gradients(zip(gradients[len(q_origin_model1.trainable_variables):], q_origin_model2.trainable_variables))


def update_target():
    for var, var_target in zip(q_origin_model1.variables, q_target_model1.variables):
        var_target.assign(sac_tau * var + (1.0 - sac_tau) * var_target)
    for var, var_target in zip(q_origin_model2.variables, q_target_model2.variables):
        var_target.assign(sac_tau * var + (1.0 - sac_tau) * var_target)







def main():
    

    # pi_model.model.load_weights('./SAC/my_checkpoint_Actor0')
    reward_records = []
    all_ig = []
    all_step_cost = []
    all_cost = []
    all_sim = []
    all_pv = []
    all_count = []
    action_mean = []
    action_std = []
    action_max = []
    action_min = []

    
    
    t = tqdm.tqdm(range(Trainig_Episodes_NO))
    for i in t:
        env_number = 1 
        pher_condition = False
        map_generator = Generator(np.random.randint(0,100000))
        myenv = env(map_generator.get_map(),env_number,pher_condition)
        s , x , m , localx , localy,pher_map = myenv.reset()
        done = False
        cum_reward = 0
    
        episode_step = 0
        ig = 0
        sc = 0
        pv = 0
        action_hist = []

        while not done:
            episode_step += 1
            a = pick_sample(s)
            action_hist.append(a)

            s_next, x , m , localx , localy , done ,r , plotinfo, pher_map = myenv.step(a,x,m,localx,localy,pher_map)



            if episode_step >= maxstep or done==True:
                done = True
                reward , totallen , sim  = myenv.finish_reward(m,localx,localy,episode_step)
                r += reward
            

            buffer.add([s.tolist(), a, r, s_next.tolist(), float(done)])
            cum_reward += r
            ig += plotinfo[0]
            sc += plotinfo[1]
            pv += plotinfo[2]

            if buffer.length() >= sac_batch_size or done==True:
                states, actions, rewards, n_states, dones = buffer.sample(sac_batch_size)
                optimize_theta(states)
                optimize_phi(states, actions, rewards, n_states, dones)
                update_target()

            s = s_next   

        reward_records.append(cum_reward)
        all_ig.append(ig)
        all_step_cost.append(sc)
        all_cost.append(totallen)
        all_sim.append(sim)
        all_pv.append(pv)
        all_count.append(episode_step)
        action_mean.append(np.mean(action_hist))
        action_std.append(np.std(action_hist))
        action_max.append(np.max(action_hist))
        action_min.append(np.min(action_hist))
        

        if i % 50 == 0:
            pi_model.model.save_weights('./SAC/my_checkpoint_Actor{}_map'.format(i))
            q_origin_model1.model.save_weights('./SAC/my_checkpoint_Critic1{}_map'.format(i))
            q_origin_model2.model.save_weights('./SAC/my_checkpoint_Critic2{}_map'.format(i))
            dict = {'Episode Reward': reward_records,'Information Gain':all_ig,'Pheromone value':all_pv,'Step Cost':all_step_cost,'Cost':all_cost,
                    'Similarity':all_sim,'Count':all_count, 'Action Mean':action_mean,'Action Std':action_std,
                    'Action Max':action_max, 'Action Min':action_min}
            df = pd.DataFrame(dict) 
            df.to_csv('SAC_Info.csv')


            np.save('./maps/SAC_map_{}.npy'.format(i),m)
            np.save('./maps/SAC_localx_{}.npy'.format(i),localx)
            np.save('./maps/SAC_localy_{}.npy'.format(i),localy)
            # plt.imshow(np.subtract(1,m), cmap='gray', origin='lower', vmin=0.0, vmax=1.0)
            # plt.plot(localx[0], localy[0], 'ro')
            # plt.plot(localx, localy)
            # plt.show()


env_number = 1 
pher_condition = False
map_generator = Generator(np.random.randint(0,100000))
myenv = env(map_generator.get_map(),env_number,pher_condition)
buffer = ReplayBuffer(10000)
state_dim = myenv.state_dim
action_dim = myenv.action_dim

pi_model = PolicyNet(state_dim, action_dim)

q_origin_model1 = QNet(state_dim, action_dim)  
q_origin_model2 = QNet(state_dim, action_dim)  
q_target_model1 = QNet(state_dim, action_dim)  
q_target_model2 = QNet(state_dim, action_dim)  

opt_pi = tf.keras.optimizers.Adam(learning_rate = sac_lr)
opt_q1 = tf.keras.optimizers.Adam(learning_rate = 5*sac_lr)
opt_q2 = tf.keras.optimizers.Adam(learning_rate = 5*sac_lr)

def eval_agent(test_env,map_matrix):

    from utils import sim_map,total_len
    TotalPath_length , all_sim , all_len, all_episode_reward  , action_hist , plot_sim , plot_len, Topo_length = [], [],[],[],[],[],[],[]

    state_dim = test_env.state_dim
    action_dim = test_env.action_dim

    pi_model = PolicyNet(state_dim, action_dim)



    pi_model.model.load_weights('./SAC/my_checkpoint_Actor14950_map')
    s , x , m , localx , localy,pher_map = test_env.reset()
    done = False
    cum_reward = 0

    episode_step = 0
    

    while not done:
        episode_step += 1
        a = pick_sample(s)
        action_hist.append(a)
        s_next, x , m , localx , localy , done ,r , plotinfo, pher_map = test_env.step(a,x,m,localx,localy,pher_map)


        if episode_step >= 200 or done==True:
            done = True
            reward , totallen , sim  = test_env.finish_reward(m,localx,localy,episode_step)
            r += reward
        

        if sim_map(map_matrix,m) >= 0.7 and sim_map(map_matrix,m) <= 0.9 :
            Topo_length.append(total_len(localx,localy))
        cum_reward += r
        plot_sim.append(sim_map(map_matrix,m))
        plot_len.append(total_len(localx,localy))
        
        s = s_next
        episode_step += 1

    TotalPath_length.append(totallen)
    all_sim.append(plot_sim)
    all_len.append(plot_len)
    all_episode_reward.append(cum_reward)

    return action_hist, TotalPath_length , all_sim , all_len, all_episode_reward , localx , localy,Topo_length


if __name__ == "__main__":
    # env_number = 1 
    # pher_condition = False
    # map_generator = Generator(np.random.randint(0,100000))
    # myenv = env(map_generator.get_map(),env_number,pher_condition)
    # buffer = ReplayBuffer(10000)
    # state_dim = myenv.state_dim
    # action_dim = myenv.action_dim

    # pi_model = PolicyNet(state_dim, action_dim)

    # q_origin_model1 = QNet(state_dim, action_dim)  
    # q_origin_model2 = QNet(state_dim, action_dim)  
    # q_target_model1 = QNet(state_dim, action_dim)  
    # q_target_model2 = QNet(state_dim, action_dim)  

    # opt_pi = tf.keras.optimizers.Adam(learning_rate = sac_lr)
    # opt_q1 = tf.keras.optimizers.Adam(learning_rate = 5*sac_lr)
    # opt_q2 = tf.keras.optimizers.Adam(learning_rate = 5*sac_lr)

    main()