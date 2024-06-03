import tensorflow as tf
from tensorflow.keras.layers import ( Input, Conv2D, Dense, Flatten,Dropout )
tf.keras.backend.set_floatx('float64')
#import matplotlib.pyplot as plt 
import numpy as np
from environment import env
import warnings
import logging
warnings.filterwarnings("ignore")
tf.get_logger().setLevel(logging.ERROR)
import pandas as pd
from threading import Thread, Lock
from map_generator import Map_Generator as Generator
from config import (ppo_actor_lr ,ppo_entropy_coefficent ,ppo_clip_ratio
                   ,ppo_batch_size ,ppo_n_epochs,Trainig_Episodes_NO ,gamma
                   ,ppo_lmbda ,ppo_n_workers,maxep,maxstep)
from tqdm import tqdm
import os
import glob
# import matplotlib.pyplot as plt


from utils import(frontier_space)

CUR_EPISODE, all_rewards, all_cost, all_sim, all_count, all_ig,all_step_cost, all_pv, actor_loss_list, policy_loss_list, entropy_list, critic_loss_list , action_mean , action_std  = 0, [], [], [],[], [], [], [], [], [], [] , [] , [] ,[] 

icm_lr = 1e-4

class ICM:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        # self.model = self.create_model()
        self.cnn = self.create_cnn()
        self.inv_model = self.inverse_model()
        self.forw_model = self.forward_model()
        self.opt = tf.keras.optimizers.Adam(icm_lr, epsilon=1e-05)

    def create_cnn(self):
        return tf.keras.Sequential([
            Input((self.state_dim[0] , self.state_dim[1] , self.state_dim[2])),
            (Conv2D(32, kernel_size=(8,8), strides=(4,4), activation='relu')),
            # Dropout(0.2),
            (Conv2D(64 , kernel_size=(4,4), strides=(2,2), activation='relu')),
            # Dropout(0.2),
            (Conv2D(128 , kernel_size=(4,4), strides=(2,2), activation='relu')),
            # Dropout(0.2),
            (Flatten()),
        ])
    
    def inverse_model(self):
        # cnn = self.create_cnn()
        input1 = Input(shape=(self.state_dim[0], self.state_dim[1], self.state_dim[2]))
        input2 = Input(shape=(self.state_dim[0], self.state_dim[1], self.state_dim[2]))
        feature1 = self.cnn(input1)
        feature2 = self.cnn(input2)
        combinedInput = tf.keras.layers.concatenate([feature1, feature2])
        x = Dense(512, activation='relu')(combinedInput)
        x = Dense(self.action_dim, activation='softmax')(x)
        model = tf.keras.Model(inputs=[input1,input2], outputs=x)
        return model
    
    def forward_model(self):
        # cnn = self.create_cnn()
        input1 = Input(shape=(self.state_dim[0], self.state_dim[1], self.state_dim[2]))
        feature1 = self.cnn(input1)
        action_input = Input(shape=(1,), name='action_input')
        combinedInput = tf.keras.layers.concatenate([feature1,action_input])
        x = Dense(512, activation='relu')(combinedInput) 
        x = Dense(1, activation='relu')(x)
        model = tf.keras.Model(inputs=[input1,action_input], outputs=x)
        return model

    
    
    
    
    def compute_inv_loss(self, pred, action):
        # Calculate the depth of the one-hot encoding
        # pred = tf.reshape(pred, [len(pred), 1])
        depth = self.action_dim

        action = tf.one_hot(action, depth)
        # pred = tf.one_hot(pred, depth)

        pred = tf.cast(pred, tf.float64)
        action = tf.cast(action, tf.float64) 

        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True,label_smoothing = 0.1)
        # print("inv loss is: ",loss(action, pred))

        return loss(action, pred)

    
    def compute_forward_loss(self, pred, next_state):
        next_state = tf.cast(next_state, tf.float64)
        pred = tf.cast(pred, tf.float64)

        loss = tf.keras.losses.MeanSquaredError()
        # print("The other loss is: ",loss(next_state,pred))
        return loss(next_state,pred)

    def train(self, state, next_state, action):
        with tf.device('/cpu:0'):
            with tf.GradientTape(persistent=True) as tape:
                state = np.reshape(state, [1, self.state_dim[0],self.state_dim[1],self.state_dim[2]])
                next_state = np.reshape(next_state, [1, self.state_dim[0],self.state_dim[1],self.state_dim[2]])
                pred_action_probs = self.inv_model([state, next_state],training=True)
                pred_next_state = self.forw_model([state, action],training=True)
                next_feature_vector = self.cnn(np.reshape(next_state, [1, self.state_dim[0],self.state_dim[1],self.state_dim[2]]),training=False)
                forward_loss = self.compute_forward_loss(pred_next_state, next_feature_vector)
                inv_loss = self.compute_inv_loss(pred_action_probs, action)
            grads_encoder = tape.gradient(inv_loss, self.inv_model.trainable_variables + self.cnn.trainable_variables)
            self.opt.apply_gradients(zip(grads_encoder, self.inv_model.trainable_variables + self.cnn.trainable_variables))
            grads_forward = tape.gradient(forward_loss,self.forw_model.trainable_variables)
            self.opt.apply_gradients(zip(grads_forward, self.forw_model.trainable_variables))
        return inv_loss , forward_loss




class Actor:
    def __init__(self, state_dim, action_dim,actor_lr,entropy_beta,clip_ratio):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = self.create_model()
        self.entropy_beta = entropy_beta
        self.actor_lr = actor_lr 
        self.clip_ratio = clip_ratio
        self.opt = tf.keras.optimizers.Adam(self.actor_lr,epsilon=1e-03)
    
    def create_cnn(self):
        return tf.keras.Sequential([
            Input((self.state_dim[0] , self.state_dim[1] , self.state_dim[2])),
            (Conv2D(32, kernel_size=(8,8), strides=(4,4), activation='relu')),
            # Dropout(0.2),
            (Conv2D(64 , kernel_size=(4,4), strides=(2,2), activation='relu')),
            # Dropout(0.2),
            (Conv2D(128 , kernel_size=(4,4), strides=(2,2), activation='relu')),
            # Dropout(0.2),
            (Flatten()),
        ])
    
    def create_model(self):
        cnn = self.create_cnn()
        cnn.build((None, self.state_dim[0] , self.state_dim[1] , self.state_dim[2]))
        y = Dense(512, activation='relu')(cnn.output)
        # y = Dropout(0.2)(y)
        y = Dense(self.action_dim, activation='softmax')(y)
        model = tf.keras.Model(inputs=[cnn.input], outputs=y)
        return model



    def compute_loss(self, old_policy, new_policy, actions, gaes):
        gaes = tf.stop_gradient(gaes)
        old_log_p = tf.math.log(
            tf.reduce_sum(old_policy * actions))
        old_log_p = tf.stop_gradient(old_log_p)
        log_p = tf.math.log(tf.reduce_sum(
            new_policy * actions))
        ratio = tf.math.exp(log_p - old_log_p)
        clipped_ratio = tf.clip_by_value(
            ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        surrogate = -tf.minimum(ratio * gaes, clipped_ratio * gaes)
        policy_loss = tf.reduce_mean(surrogate)
        entropy_loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits = True)
        entropy = entropy_loss(new_policy,new_policy)
        total_loss = policy_loss - self.entropy_beta*entropy 
        return total_loss , policy_loss , entropy

    def train(self, old_policy, states, actions, gaes):
        actions = tf.one_hot(actions, self.action_dim)
        actions = tf.reshape(actions, [-1, self.action_dim])
        actions = tf.cast(actions, tf.float64)

        with tf.device('/cpu:0'):
            with tf.GradientTape() as tape:
                state_res = np.reshape(states, [len(states), self.state_dim[0],self.state_dim[1],self.state_dim[2]])
                logits = self.model([state_res], training=True)
                loss , policy_loss , entropy = self.compute_loss(old_policy, logits, actions, gaes)
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss , policy_loss , entropy


class Critic:
    def __init__(self, state_dim, critic_lr):
        self.state_dim = state_dim
        self.model = self.create_model()
        self.critic_lr  = critic_lr
        self.opt = tf.keras.optimizers.Adam(self.critic_lr, epsilon=1e-03) 
    
    def create_cnn(self):
        return tf.keras.Sequential([
            Input((self.state_dim[0] , self.state_dim[1] , self.state_dim[2])),
            (Conv2D(32, kernel_size=(8,8), strides=(4,4), activation='relu')),
            Dropout(0.2),
            (Conv2D(64 , kernel_size=(4,4), strides=(2,2), activation='relu')),
            # Dropout(0.2),
            (Conv2D(128 , kernel_size=(4,4), strides=(2,2), activation='relu')),
            # Dropout(0.2),
            (Flatten()),
        ])
    
    def create_model(self):
        cnn = self.create_cnn()
        cnn.build((None, self.state_dim[0] , self.state_dim[1] , self.state_dim[2]))
        y = Dense(512, activation='relu')(cnn.output)
        # y = Dropout(0.2)(y)
        y = Dense(1, activation='linear')(y)
        model = tf.keras.Model(inputs=[cnn.input], outputs=y)
        return model


    def compute_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    def train(self, states, td_targets):
        with tf.device('/cpu:0'):
            with tf.GradientTape() as tape:
                v_pred = self.model([np.reshape(states, [len(states), self.state_dim[0],self.state_dim[1],self.state_dim[2]])],training=True)
                assert v_pred.shape == td_targets.shape
                loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss



class PPO_Agent:
    def __init__(self, actor_lr, entropy_beta, clip_ratio, critic_lr,
                 gamma,lmbda, batch_size, n_epochs, hyperparameter_name,value , action_setting,
                 pher_condition):
        self.env_number = action_setting
        self.pher_condtion = pher_condition
        map_generator = Generator(np.random.randint(0,100000))
        self.environment = env(map_generator.ref_map(),self.env_number,self.pher_condtion)
        self.state_dim = self.environment.state_dim
        self.action_dim = self.environment.action_dim
        self.gamma = gamma 
        self.lmbda = lmbda
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.entropy_beta = entropy_beta
        self.clip_ratio = clip_ratio
        self.hyperparameter_name = hyperparameter_name
        self.value = value
        self.icm = ICM(self.state_dim, self.action_dim)
        

        self.actor = Actor(self.state_dim, self.action_dim, self.actor_lr, self.entropy_beta, self.clip_ratio)
        self.critic = Critic(self.state_dim,self.critic_lr)

    def load_model(self,hyperparameter_name,value):

        self.actor.model.load_weights('./appo/my_checkpoint_Actor_{}_{}_env{}_icm'.format(hyperparameter_name,value,self.env_number))
        self.critic.model.load_weights('./appo/my_checkpoint_Critic_{}_{}_env{}_icm'.format(hyperparameter_name,value,self.env_number))


    def save_model(self,hyperparameter_name,value):

        self.actor.model.save_weights('./appo/my_checkpoint_Actor_{}_{}_env{}_icm'.format(hyperparameter_name,value,self.env_number))
        self.critic.model.save_weights('./appo/my_checkpoint_Critic_{}_{}_env{}_icm'.format(hyperparameter_name,value,self.env_number))

    def save_data(self,name,value):
        global CUR_EPISODE, all_rewards, all_cost, all_sim, all_count, all_ig, all_pv, actor_loss_list, policy_loss_list, entropy_list, critic_loss_list , action_mean , action_std
        dict = {'Episode Reward': all_rewards,'Information Gain':all_ig,'Pheromone value':all_pv,'Step Cost':all_step_cost,'Cost':all_cost,'Similarity':all_sim,'Count':all_count, 'Action Mean':action_mean,'Action Std':action_std}
        df = pd.DataFrame(dict)
        df.to_csv('PPO_Info_{}_{}_env_{}_pher_{}_icm.csv'.format(name,value,self.env_number,self.pher_condtion))
        
        dict ={'Actor Loss':actor_loss_list,'Policy Loss':policy_loss_list,'Entropy':entropy_list,"Critic Loss":critic_loss_list}
        df = pd.DataFrame(dict)
        df.to_csv('PPO_Loss_Info_{}_{}_env{}_pher_{}_icm.csv'.format(name,value,self.env_number,self.pher_condtion))
    
    def delete_old_checkpoints(self, hyperparameter_name, value, keep_last_n=2):
        pattern = f'./appo/my_checkpoint_*_{hyperparameter_name}_{value}.*'
        checkpoints = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        for checkpoint in checkpoints[keep_last_n:]:
            os.remove(checkpoint)

    
    def gae_target(self, rewards, v_values, next_v_value, done):
        n_step_targets = np.zeros_like(rewards)
        gae = np.zeros_like(rewards)
        gae_cumulative = 0
        forward_val = 0

        if not done:
            forward_val = next_v_value

        for k in reversed(range(0, len(rewards))):
            delta = rewards[k] + self.gamma * forward_val - v_values[k]
            gae_cumulative = self.gamma * self.lmbda * gae_cumulative + delta
            gae[k] = gae_cumulative
            forward_val = v_values[k]
            n_step_targets[k] = gae[k] + v_values[k]
        return gae, n_step_targets

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch
    
    def get_feature_vector(self, state):
        return self.icm.cnn.predict(np.reshape(state, [1, self.actor.state_dim[0],self.actor.state_dim[1],self.actor.state_dim[2]]),verbose=None)
    
    def calc_intr_reward(self, state, next_state, action):
        next_feature_vector = self.get_feature_vector(next_state) #(phi_st+1)
        pred_next_state = self.icm.forw_model([state, action])
        return self.icm.compute_forward_loss(pred_next_state, next_feature_vector)
    
    def train(self, map_number , max_ep_permap=1000, max_steps=1000,number_of_workers=1):

        global CUR_EPISODE, all_rewards, all_cost, all_sim, all_count, all_ig,all_step_cost, all_pv, actor_loss_list, policy_loss_list, entropy_list, critic_loss_list , action_mean , action_std
        map_episode  = 0 

        

        self.max_episodes = max_ep_permap
        self.max_steps = max_steps
        t = tqdm(range(self.max_episodes))

        for _ in t:
            self.env = env(Generator(np.random.randint(0,15000)).ref_map(),self.env_number,self.pher_condtion)
            self.map_number = map_number
            state_batch = []
            next_state_batch = []
            action_batch = []
            reward_batch = []
            old_policy_batch = []

            episode_reward, done = 0, False

            state , x , m , localx , localy , pher_map  = self.env.reset()


        
            reward = 0 
            action = 0 
            episode_step = 0
            ig = 0
            step_cost = 0 
            pv = 0 
            action_hist = []

            while not done:
                episode_step += 1
                with tf.device('/cpu:0'):
                    probs = self.actor.model.predict(
                        [np.reshape(state, [1, self.actor.state_dim[0],self.actor.state_dim[1],self.state_dim[2]])],verbose=None)
                
                
                action = np.random.choice(self.actor.action_dim, p=probs[0])
                action_hist.append(action)
                

                next_state, x , m , localx , localy , done ,reward  , plotinfo , pher_map = self.env.step(action,x,m,localx,localy,pher_map)

                intr_reward = self.calc_intr_reward(np.reshape(state, [1, self.actor.state_dim[0],self.actor.state_dim[1],self.state_dim[2]]), 
                                                    np.reshape(next_state, [1, self.actor.state_dim[0],self.actor.state_dim[1],self.state_dim[2]]), 
                                                    np.reshape(action,[1,1]))
                
                intr_reward = float(intr_reward)
                # print("Intrinsic reward: ",intr_reward)
                # print("Extrinsic reward: ",reward)
                reward = intr_reward


                state = np.reshape(state, [1, self.actor.state_dim[0] , self.actor.state_dim[1],self.state_dim[2]])
                action = np.reshape(action, [1, 1])
                next_state = np.reshape(next_state, [1, self.actor.state_dim[0] , self.actor.state_dim[1],self.state_dim[2]])

                if episode_step >= self.max_steps or done==True:
                    done = True
                    r , totallen , sim  = self.env.finish_reward(m,localx,localy,episode_step)
                    reward += r 
                    

                reward = np.reshape(reward, [1, 1])
                state_batch.append(state)
                next_state_batch.append(next_state)
                action_batch.append(action)
                reward_batch.append(reward)
                old_policy_batch.append(probs)

                if  len(state_batch) >= self.batch_size or done:
                    states = self.list_to_batch(state_batch)
                    actions = self.list_to_batch(action_batch)
                    rewards = self.list_to_batch(reward_batch)
                    old_policys = self.list_to_batch(old_policy_batch)
                    next_states = self.list_to_batch(next_state_batch)

                    with tf.device('/cpu:0'):
                        state_reshaped = np.reshape(states, [len(states), self.actor.state_dim[0],self.actor.state_dim[1],self.state_dim[2]])
                        v_values = self.critic.model.predict([state_reshaped],verbose=None)
                        next_v_value = self.critic.model.predict([np.reshape(next_state, [1, self.actor.state_dim[0],self.actor.state_dim[1],self.state_dim[2]])],verbose=None)

                       

                    gaes, td_targets = self.gae_target(
                        rewards, v_values, next_v_value, done)
                    

        
                    
                    for epoch in range(self.n_epochs):
                        actor_loss, policy_loss, entropy  = self.actor.train(old_policys, states, actions, gaes)
                        critic_loss = self.critic.train(states, td_targets)
                        actor_loss_list.append(float(actor_loss))
                        policy_loss_list.append(float(policy_loss))
                        entropy_list.append(float(entropy))
                        critic_loss_list.append(float(critic_loss))
                        for k in range(len(states)):
                                icminv_loss , icmfor_loss = self.icm.train(states[k], next_states[k], actions[k])
                   
                        
                        

                    state_batch = []
                    action_batch = []
                    reward_batch = []
                    old_policy_batch = []
                    next_state_batch = []

                
                episode_reward += reward[0][0]
                ig += plotinfo[0]
                step_cost += plotinfo[1]
                pv += plotinfo[2]

                state = next_state[0]

            

            all_rewards.append(episode_reward)
            all_ig.append(ig)
            all_step_cost.append(step_cost)
            all_pv.append(pv)
            all_cost.append(totallen)
            all_sim.append(sim)
            all_count.append(episode_step)
            action_mean.append(np.mean(action_hist))
            action_std.append(np.std(action_hist))
           
            

            
            CUR_EPISODE += 1
            map_episode += 1

            
            

            if CUR_EPISODE % 50 == 0:
                 np.save('./maps/map_{}_{}_{}_{}_{}.npy'.format(CUR_EPISODE,self.hyperparameter_name,self.value,self.pher_condtion,self.env_number),m)
                 np.save('./maps/localx_{}_{}_{}_{}_{}.npy'.format(CUR_EPISODE,self.hyperparameter_name,self.value,self.pher_condtion,self.env_number),localx)
                 np.save('./maps/localy_{}_{}_{}_{}_{}.npy'.format(CUR_EPISODE,self.hyperparameter_name,self.value,self.pher_condtion,self.env_number),localy)
                 self.save_model(self.hyperparameter_name,self.value)
                 self.save_data(self.hyperparameter_name,self.value)
                #  plt.imshow(np.subtract(1,m),cmap='gray', vmin=0, vmax=1, origin='lower')
                #  plt.plot(localx,localy,'r')
                #  plt.plot(x[0][1],x[0][0],'bo')
                #  plt.plot(localx[0],localy[0],'go')
                #  plt.show()




def main(hyperparameter_name,hyperparameter_values,action_setting,pher_condition):
    global ppo_actor_lr ,ppo_entropy_coefficent ,ppo_clip_ratio,ppo_batch_size ,ppo_n_epochs,Trainig_Episodes_NO ,gamma,ppo_lmbda ,ppo_n_workers,maxep,maxstep
    global CUR_EPISODE, all_rewards, all_cost, all_sim, all_count, all_ig, all_pv, actor_loss_list, policy_loss_list, entropy_list, critic_loss_list

    for value in hyperparameter_values:
        # Set the specified hyperparameter to the current value
        if hyperparameter_name == 'actor_lr':
            ppo_actor_lr = value
        elif hyperparameter_name == 'entropy_coefficient':
            ppo_entropy_coefficent = value
        elif hyperparameter_name == 'clip_ratio':
            ppo_clip_ratio = value
        elif hyperparameter_name == 'batch_size':
            ppo_batch_size = value
        elif hyperparameter_name == 'n_epochs':
            ppo_n_epochs = value
        else:
            raise ValueError("Invalid hyperparameter_name")
        # action_setting = 1 
        agent = PPO_Agent(ppo_actor_lr, ppo_entropy_coefficent, ppo_clip_ratio, 
                                5*ppo_actor_lr,gamma,ppo_lmbda, ppo_batch_size, 
                                ppo_n_epochs, hyperparameter_name , value , action_setting,pher_condition)
        total_episodes = int(Trainig_Episodes_NO/ppo_n_workers)
        agent.train(0,total_episodes,maxstep+1,ppo_n_workers)
        agent.save_model(hyperparameter_name,value)
        agent.save_data(hyperparameter_name,value)
        CUR_EPISODE, all_rewards, all_cost, all_sim, all_count, all_ig, all_pv, actor_loss_list, policy_loss_list, entropy_list, critic_loss_list = 0, [], [], [], [], [], [], [], [], [] , []

def eval_agent(test_env , map_matrix , hyperparameter_name , value):
    
    from utils import sim_map , total_len
    
    action_setting = 1
    pher_condition = False
    agent = PPO_Agent(ppo_actor_lr, ppo_entropy_coefficent, ppo_clip_ratio, 
                                5*ppo_actor_lr,gamma,ppo_lmbda, ppo_batch_size, 
                                ppo_n_epochs, hyperparameter_name , value , action_setting,pher_condition)
    agent.load_model(hyperparameter_name,value)
    episode_reward, done = 0, False
    state , x , m , localx , localy, pher_map  = test_env.reset()
    reward = 0 
    action = 0 
    episode_step = 0
    TotalPath_length , all_sim , all_len, all_episode_reward  , action_hist , plot_sim , plot_len , Topo_length = [],[],[],[],[],[],[],[]


    while not done:
        episode_step += 1
        with tf.device('/cpu:0'):
            probs = agent.actor.model.predict(
                [np.reshape(state, [1, agent.actor.state_dim[0],agent.actor.state_dim[1],1])],verbose=None)
        
        
        action = np.random.choice(agent.actor.action_dim, p=probs[0])
        action_hist.append(action)
        

        next_state, x , m , localx , localy , done ,reward  , plotinfo,pher_map = test_env.step(action,x,m,localx,localy,pher_map)

        state = np.reshape(state, [1, agent.actor.state_dim[0] , agent.actor.state_dim[1],1])
        action = np.reshape(action, [1, 1])
        next_state = np.reshape(next_state, [1, agent.actor.state_dim[0] , agent.actor.state_dim[1],1])

        if episode_step >= maxstep or done==True:
            done = True
            r , totallen , sim  = test_env.finish_reward(m,localx,localy,episode_step)
            reward += r 
        
        if sim_map(map_matrix,m) >= 0.7 and sim_map(map_matrix,m) <= 0.9 :
            Topo_length.append(total_len(localx,localy))

        episode_reward += reward
        state = next_state[0]
        plot_sim.append(sim_map(map_matrix,m))
        plot_len.append(total_len(localx,localy))

    TotalPath_length.append(totallen)
    all_sim.append(plot_sim)
    all_len.append(plot_len)
    all_episode_reward.append(episode_reward)
    

    return action_hist, TotalPath_length , all_sim , all_len, all_episode_reward , localx , localy, Topo_length



if __name__ == '__main__':
    hyperparameter_name = 'actor_lr'
    hyperparameter_values = [1e-4]
    action_setting = 1
    pher_condition = False
    main(hyperparameter_name,hyperparameter_values,action_setting, pher_condition)