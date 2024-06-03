import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import random
import math
from torch.utils.tensorboard import SummaryWriter
from collections import deque, namedtuple
import time
import gym
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def weight_init(layers):
    for layer in layers:
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')


class IQN(nn.Module):
    def __init__(self, state_size, action_size,layer_size, n_step, seed, layer_type="ff"):
        super(IQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_shape = state_size
        self.action_size = action_size
        self.K = 8
        self.N = 16
        self.n_cos = 64
        self.layer_size = layer_size
        self.pis = torch.FloatTensor([np.pi*i for i in range(self.n_cos)]).view(1,1,self.n_cos).to(device) # Starting from 0 as in the paper 

        # self.head = nn.Sequential(
        #         nn.Conv2d(self.input_shape[2], out_channels=32, kernel_size=8, stride=4),
        #         nn.ReLU(),
        #         nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
        #         nn.ReLU(),
        #         nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
        #         nn.ReLU(),
        #         # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
        #     )#.apply() #weight init

        self.head = nn.Sequential(
                nn.Conv2d(self.input_shape[2], out_channels=32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=512, kernel_size=4, stride=2),
                nn.ReLU(),
                # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            )#.apply() #weight init
        
        

        
        self.cos_embedding = nn.Linear(self.n_cos, layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.ff_2 = nn.Linear(layer_size, action_size)
        self.advantage = nn.Linear(layer_size, action_size)
        self.value = nn.Linear(layer_size, 1)
        #weight_init([self.head_1, self.ff_1])


        
    def calc_cos(self, batch_size, n_tau=8, Cvar = 1):
        """
        Calculating the cosinus values depending on the number of tau samples
        """
        taus = torch.rand(batch_size, n_tau).to(device).unsqueeze(-1) #(batch_size, n_tau, 1)
        taus = taus*Cvar
        cos = torch.cos(taus*self.pis)

        assert cos.shape == (batch_size,n_tau,self.n_cos), "cos shape is incorrect"
        return cos, taus
    
    def forward(self, input, num_tau=8, Cvar=1):
        """
        Quantile Calculation depending on the number of tau
        
        Return:
        quantiles [ shape of (batch_size, num_tau, action_size)]
        taus [shape of ((batch_size, num_tau, 1))]
        
        """
        batch_size = input.shape[0]
        
        x = torch.relu(self.head(input))
        x = x.view(input.size(0), -1) 
        cos, taus = self.calc_cos(batch_size, num_tau,Cvar) # cos shape (batch, num_tau, layer_size)
        cos = cos.view(batch_size*num_tau, self.n_cos)
        cos_x = torch.relu(self.cos_embedding(cos)).view(batch_size, num_tau, self.layer_size) # (batch, n_tau, layer)
        
        # x has shape (batch, layer_size) for multiplication –> reshape to (batch, 1, layer)
        x = (x.unsqueeze(1)*cos_x).view(batch_size*num_tau, self.layer_size)
        
        x = torch.relu(self.ff_1(x))
        # out = self.ff_2(x)
        advantage = self.advantage(x)
        value = self.value(x)
        out = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return out.view(batch_size, num_tau, self.action_size), taus
    
    def get_action(self, inputs,Cvar=1):
        quantiles, _ = self.forward(inputs, self.K,Cvar)
        actions = quantiles.mean(dim=1)
        return actions


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, device, seed, gamma, n_step=1):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = device
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=self.n_step)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        #print("before:", state,action,reward,next_state, done)
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) == self.n_step:
            state, action, reward, next_state, done = self.calc_multistep_return()
            #print("after:",state,action,reward,next_state, done)
            e = self.experience(state, action, reward, next_state, done)
            self.memory.append(e)
    
    def calc_multistep_return(self):
        Return = 0
        for idx in range(self.n_step):
            Return += self.gamma**idx * self.n_step_buffer[idx][2]
        
        return self.n_step_buffer[0][0], self.n_step_buffer[0][1], Return, self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
        
    
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    

class DQN_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 layer_size,
                 n_step,
                 BATCH_SIZE,
                 BUFFER_SIZE,
                 LR,
                 TAU,
                 GAMMA,
                 UPDATE_EVERY,
                 device,
                 seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            layer_size (int): size of the hidden layer
            BATCH_SIZE (int): size of the training batch
            BUFFER_SIZE (int): size of the replay memory
            LR (float): learning rate
            TAU (float): tau for soft updating the network weights
            GAMMA (float): discount factor
            UPDATE_EVERY (int): update frequency
            device (str): device that is used for the compute
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = device
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.UPDATE_EVERY = UPDATE_EVERY
        self.BATCH_SIZE = BATCH_SIZE
        self.Q_updates = 0
        self.n_step = n_step

        self.action_step = 4
        self.last_action = None

        # IQN-Network
        self.qnetwork_local = IQN(state_size, action_size,layer_size, n_step, seed).to(device)
        self.qnetwork_target = IQN(state_size, action_size,layer_size, n_step, seed).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        # print(self.qnetwork_local)
        
        
        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, self.device, seed, self.GAMMA, n_step)
        
        # self.memory = PrioritizedReplay
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done, writer):
        # add to the dimension of the state
        state = np.expand_dims(state, axis=0)
        next_state = np.expand_dims(next_state, axis=0)
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()
                loss = self.learn(experiences)
                self.Q_updates += 1
                writer.add_scalar("Q_loss", loss, self.Q_updates)

    def act(self, state, eps=0., Cvar=1):
        """Returns actions for given state as per current policy. Acting only every 4 frames!
        
        Params
        ======
            frame: to adjust epsilon
            state (array_like): current state
            
        """

        state = np.array(state)

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local.get_action(state,Cvar)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps: # select greedy action if random number is higher than epsilon or noisy network is used!
            action = np.argmax(action_values.cpu().data.numpy())
            self.last_action = action
            return action
        else:
            action = random.choice(np.arange(self.action_size))
            self.last_action = action 
            return action

        # else:
        #     self.action_step += 1
        #     return self.last_action


    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones = experiences
        # print("states ", states.shape)
        # print("actions ", actions.shape)
        # print("rewards ", rewards.shape)
        # print("next_states ", next_states.shape)
        # print("dones ", dones.shape)

        # print(states.shape, actions.shape, rewards.shape, next_states.shape, dones.shape)
        # Get max predicted Q values (for next states) from target model
        Q_targets_next, _ = self.qnetwork_target(next_states)
        Q_targets_next = Q_targets_next.detach().max(2)[0].unsqueeze(1) # (batch_size, 1, N)
        
        # Compute Q targets for current states 
        Q_targets = rewards.unsqueeze(-1) + (self.GAMMA**self.n_step * Q_targets_next * (1. - dones.unsqueeze(-1)))
        # Get expected Q values from local model
        Q_expected, taus = self.qnetwork_local(states)
        Q_expected = Q_expected.gather(2, actions.unsqueeze(-1).expand(self.BATCH_SIZE, 8, 1)) # what does this do? 

        # Quantile Huber loss
        td_error = Q_targets - Q_expected
        assert td_error.shape == (self.BATCH_SIZE, 8, 8), "wrong td error shape"
        huber_l = calculate_huber_loss(td_error, 1.0)
        quantil_l = abs(taus -(td_error.detach() < 0).float()) * huber_l / 1.0
        
        loss = quantil_l.sum(dim=1).mean(dim=1) # , keepdim=True if per weights get multipl
        loss = loss.mean()


        # Minimize the loss
        loss.backward()
        #clip_grad_norm_(self.qnetwork_local.parameters(),1)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        return loss.detach().cpu().numpy()            

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.TAU*local_param.data + (1.0-self.TAU)*target_param.data)


def calculate_huber_loss(td_errors, k=1.0):
    """
    Calculate huber loss element-wisely depending on kappa k.
    """
    loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
    assert loss.shape == (td_errors.shape[0], 8, 8), "huber loss has wrong shape"
    return loss

from config import (map_height, map_width, action_size,
                    maxep,maxstep,gamma,Trainig_Episodes_NO,
                    D3QN_lr,D3QN_batch_size,D3QN_eps,D3QN_eps_decay,D3QN_eps_min,
                    D3QN_buffer_size)

from environment import env
from map_generator import Map_Generator as Generator
import warnings
import logging
warnings.filterwarnings("ignore")
CUR_EPISODE, all_rewards, all_cost, all_sim, all_count, all_ig, all_step_cost, all_pv, eps_value, action_mean , action_std , action_max , action_min  = 0, [], [], [],[], [], [], [], [] , [] , [], [],[]
from tqdm import tqdm

def train(max_episodes=10000, eps_fixed=False, eps_frames=1e6, min_eps=0.01):
    if eps_fixed:
        eps = 0
    else:
        eps = 1
    eps_start = 1
    epsilon_decay_rate  = 0.999
    i_episode = 1
    t = tqdm(range(max_episodes))
    env_number = 1
    pher_condtion = False
    for ep in t:
        myenv = env(Generator(np.random.randint(0,100000)).ref_map(),env_number,pher_condtion)
        episode_reward, done = 0, False
        state , x , m , localx , localy, pher_map = myenv.reset()

        reward = 0 
        action = 0 
        episode_step = 0
        ig = 0
        step_cost =0
        pv = 0 
        action_hist = [] 

        while not done:
            # print(eps)
            action = agent.act(state, eps)
            action_hist.append(action)
            # print(action)
            next_state, x , m , localx , localy , done ,reward , plotinfo, pher_map = myenv.step(action,x,m,localx,localy, pher_map)
            if episode_step >= 40 or done==True:
                done = True
                r , totallen , sim  = myenv.finish_reward(m,localx,localy,episode_step)
                reward += r 
            # print("state", state.shape)
            # state = np.reshape(state, [1, state_size[0], state_size[1], state_size[2]])
            # next_state = np.reshape(next_state, [1, state_size[0], state_size[1], state_size[2]])
            agent.step(state, action, reward, next_state, done, writer)
            episode_reward += reward
            state = next_state
            # state = np.reshape(state, [state_size[0], state_size[1], state_size[2]])

            ig += plotinfo[0]
            step_cost += plotinfo[1]
            pv += plotinfo[2]
            # eps = max(eps_start - (ep*(1/max_episodes)), min_eps)
            eps = eps * epsilon_decay_rate
            eps = max(eps, min_eps)

            
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
        
        if ep % 10 == 0:

            dict = {'Episode Reward': all_rewards,'Information Gain':all_ig,'Pheromone value':all_pv,'Step Cost':all_step_cost,'Cost':all_cost,
                'Similarity':all_sim,'Count':all_count, 'Action Mean':action_mean,'Action Std':action_std,
                'Action Max':action_max, 'Action Min':action_min}
            df = pd.DataFrame(dict) 
            df.to_csv('IQN_Info.csv')

        torch.save(agent.qnetwork_local.state_dict(), "IQN"+".pth")


def linear_alpha(b):
    b_start = 0.2
    b_end = 0.95
    alpha_start = 1.0
    alpha_end = 0.1
    b = max(min(b, b_end), b_start)
    alpha = alpha_start + (alpha_end - alpha_start) * ((b - b_start) / (b_end - b_start))
    alpha = max(min(alpha, alpha_start), alpha_end)
    
    return alpha

def exponential_decay_alpha(b):
    b_min=0.2
    b_max=0.95 
    alpha_start=1.0 
    alpha_min=0.1
    b = np.clip(b, b_min, b_max)
    decay_constant = np.log(alpha_min / alpha_start) / (b_max - b_min)
    alpha = alpha_start * np.exp(decay_constant * (b - b_min))
    alpha = max(alpha, alpha_min)
    
    return alpha


def concave_exponential_decay_alpha(b):
    b_min=0.2
    b_max=0.95 
    alpha_start=1.0 
    alpha_min=0.1
    b = np.clip(b, b_min, b_max)
    decay_constant = np.log(alpha_start / alpha_min) / (b_max - b_min)
    alpha = alpha_start + 1* (-b**2) * (alpha_start - alpha_min)

    
    return alpha



def eval_agent(test_env, map_matrix , Cvar , type="Fix"):
    from utils import sim_map, total_len
    from config import maxstep
    writer = SummaryWriter("runs/"+"IQN_CP_5")
    
    action_size = test_env.action_dim
    state_size = test_env.state_dim
    state_size = [state_size[0],state_size[1],state_size[2]]

    agent = DQN_Agent(state_size=state_size,    
                        action_size=action_size,
                        layer_size=512,
                        n_step=2,
                        BATCH_SIZE=100, 
                        BUFFER_SIZE=1000000, 
                        LR=5*1e-4, 
                        TAU=0.001, 
                        GAMMA=0.99, 
                        UPDATE_EVERY=16, 
                        device="cpu", 
                        seed=5)

    agent.qnetwork_local.load_state_dict(torch.load('IQN.pth'))

    IQN_action_hist = []
    IQN_TotalPath_length = []
    IQN_all_sim = []
    IQN_all_len = []
    IQN_episode_reward = []
    IQN_plot_sim = []   
    IQN_plot_len = []
    Topo_length = []
    
    
    episode_reward, done = 0, False
    state , x , m , localx , localy, pher_map = test_env.reset()
    episode_step = 0
    reward = 0 
    action = 0 

    



    while not done:
        if type == "Linear":
            Cvar = linear_alpha(sim_map(map_matrix,m))
        elif type == "Exponential":
            Cvar = exponential_decay_alpha(sim_map(map_matrix,m))
        elif type == "Concave":
            Cvar = concave_exponential_decay_alpha(sim_map(map_matrix,m))
        episode_step += 1

        eps = 0 
        action = agent.act(state, eps,Cvar)
        IQN_action_hist.append(action)
        next_state, x , m , localx , localy , done ,reward , plotinfo, pher_map = test_env.step(action,x,m,localx,localy, pher_map)
        if episode_step >= maxstep or done==True:
            done = True
            r , totallen , sim  = test_env.finish_reward(m,localx,localy,episode_step)
            reward += r 
        agent.step(state, action, reward, next_state, done, writer)
        episode_reward += reward
        state = next_state
        IQN_plot_sim.append(sim_map(map_matrix,m))
        IQN_plot_len.append(total_len(localx,localy))
        if sim_map(map_matrix,m) >= 0.7 and sim_map(map_matrix,m) <= 0.9 :
            Topo_length.append(total_len(localx,localy))    
        
    
    
    IQN_TotalPath_length.append(totallen)
    IQN_all_sim.append(IQN_plot_sim)
    IQN_all_len.append(IQN_plot_len)
    IQN_episode_reward.append(episode_reward)
    IQN_localx , IQN_localy = localx , localy

    return IQN_action_hist, IQN_TotalPath_length, IQN_all_sim, IQN_all_len, IQN_episode_reward, IQN_localx, IQN_localy, Topo_length




if __name__ == "__main__":

    writer = SummaryWriter("runs/"+"IQN_CP_5")
    seed = 5
    BUFFER_SIZE = 1000000
    BATCH_SIZE = 100
    GAMMA = 0.99
    TAU = 0.001
    LR = 5*1e-4
    UPDATE_EVERY = 16
    n_step = 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("Using ", device)




    np.random.seed(seed)
    env_number = 1
    pher_condtion = False
    myenv = env(Generator(np.random.randint(0,100000)).ref_map(),env_number,pher_condtion)
    action_size = myenv.action_dim
    state_size = myenv.state_dim
    state_size = [state_size[0],state_size[1],state_size[2]]

    agent = DQN_Agent(state_size=state_size,    
                        action_size=action_size,
                        layer_size=512,
                        n_step=n_step,
                        BATCH_SIZE=BATCH_SIZE, 
                        BUFFER_SIZE=BUFFER_SIZE, 
                        LR=LR, 
                        TAU=TAU, 
                        GAMMA=GAMMA, 
                        UPDATE_EVERY=UPDATE_EVERY, 
                        device=device, 
                        seed=seed)



    # set epsilon frames to 0 so no epsilon exploration
    eps_fixed = False

    t0 = time.time()
    final_average100 = train( 600000, eps_fixed, 5000, 0.025)
    t1 = time.time()
    
    # print("Training time: {}min".format(round((t1-t0)/60,2)))
    torch.save(agent.qnetwork_local.state_dict(), "IQN"+".pth")