import numpy as np
import tensorflow as tf
import pandas as pd
import os
from tqdm import tqdm
import argparse

# Import the PPO agent
from agents.PPO import PPOAgent

# Set TensorFlow to use float64 by default
tf.keras.backend.set_floatx('float64')

# Suppress TensorFlow warnings
import warnings
import logging
warnings.filterwarnings("ignore")
tf.get_logger().setLevel(logging.ERROR)

class ExperienceBuffer:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.clear()
    
    def clear(self):
        self.state_batch = []
        self.action_batch = []
        self.reward_batch = []
        self.next_state_batch = []
        self.done_batch = []
    
    def add(self, state, action, reward, next_state, done):
        self.state_batch.append(state)
        self.action_batch.append(action)
        self.reward_batch.append(reward)
        self.next_state_batch.append(next_state)
        self.done_batch.append(done)
    
    def is_full(self):
        return len(self.state_batch) >= self.batch_size
    
    def get_batch(self):
        return (
            self.state_batch,
            self.action_batch,
            self.reward_batch,
            self.next_state_batch,
            self.done_batch
        )


class Trainer:
    
    def __init__(self, env_creator, agent_params, training_params):
        self.env_creator = env_creator
        self.agent_params = agent_params
        self.training_params = training_params
        
        self.env = self.env_creator()
        
        self.agent = PPOAgent(
            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim,
            **agent_params
        )
        
        self.buffer = ExperienceBuffer(self.agent.batch_size)
        
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_info_gain': [],
            'episode_step_costs': [],
            'episode_costs': [],
            'episode_similarities': [],
            'action_means': [],
            'action_stds': []
        }
    
    def save_metrics(self, filename):
        df = pd.DataFrame(self.metrics)
        df.to_csv(filename, index=False)
    
    def save_agent(self, actor_path, critic_path):
        self.agent.save_models(actor_path, critic_path)
    
    def load_agent(self, actor_path, critic_path):
        self.agent.load_models(actor_path, critic_path)
    
    def train(self):
        max_episodes = self.training_params.get('max_episodes', 1000)
        max_steps = self.training_params.get('max_steps', 1000)
        save_freq = self.training_params.get('save_freq', 50)
        save_dir = self.training_params.get('save_dir', './models')
        save_maps = self.training_params.get('save_maps', True)
        maps_dir = self.training_params.get('maps_dir', './maps')
        
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(maps_dir, exist_ok=True)
        
        progress_bar = tqdm(range(max_episodes))
        
        for episode in progress_bar:
            # state, x, m, localx, localy, pher_map = self.env.reset()
            state , info = self.env.reset()
            x = info['robot_position']
            m = info['map']
            localx = info['path_x']
            localy = info['path_y']
            pher_map = info['pheromone_map']
            
            # Episode variables
            episode_reward = 0
            episode_step = 0
            episode_info_gain = 0
            episode_step_cost = 0
            episode_pher_value = 0
            action_history = []
            done = False
            
            self.buffer.clear()
            
            while not done:
                episode_step += 1
                
                action = self.agent.get_action(state)
                action_history.append(action)
                
                # next_state, x, m, localx, localy, done, reward, plotinfo, pher_map = self.env.step(
                #     action, x, m, localx, localy, pher_map
                # )

                next_state, reward, done, info = self.env.step(state, action, x, m, localx, localy, pher_map)
        
                x = info['robot_position']
                m = info['map']
                localx = info['path_x']
                localy = info['path_y']
                pher_map = info['pheromone_map']
                episode_info_gain += info['information_gain']
                episode_step_cost += info['path_cost']
                episode_pher_value += info['pheromone_value']
                
                if episode_step >= max_steps or done:
                    done = True
                    r, info= self.env.compute_final_reward()
                    totallen = info['total_path_length']
                    sim = info['map_similarity']
                    reward += r
                
                state_reshaped = np.reshape(state, [1, self.env.state_dim[0], self.env.state_dim[1], self.env.state_dim[2]])
                next_state_reshaped = np.reshape(next_state, [1, self.env.state_dim[0], self.env.state_dim[1], self.env.state_dim[2]])
                self.buffer.add(state_reshaped, action, reward, next_state_reshaped, done)
                
                episode_reward += reward
                
                
                if self.buffer.is_full() or done:
                    self.buffer.clear()
                
                state = next_state
            
            self.metrics['episode_rewards'].append(float(episode_reward))
            self.metrics['episode_lengths'].append(episode_step)
            self.metrics['episode_info_gain'].append(episode_info_gain)
            self.metrics['episode_step_costs'].append(episode_step_cost)
            self.metrics['episode_costs'].append(totallen)
            self.metrics['episode_similarities'].append(sim)
            self.metrics['action_means'].append(np.mean(action_history))
            self.metrics['action_stds'].append(np.std(action_history))
            
            progress_bar.set_description(
                f"Episode {episode} | Reward: {episode_reward:.2f} | Steps: {episode_step} | Similarity: {sim:.2f}"
            )
            
            if episode % save_freq == 0:
                self.save_agent(
                    f"{save_dir}/actor_episode_{episode}.weights.h5",
                    f"{save_dir}/critic_episode_{episode}.weights.h5"
                )
                
                self.save_metrics(f"{save_dir}/metrics_episode_{episode}.csv")
                
                if save_maps:
                    np.save(f"{maps_dir}/map_episode_{episode}.npy", m)
                    np.save(f"{maps_dir}/localx_episode_{episode}.npy", localx)
                    np.save(f"{maps_dir}/localy_episode_{episode}.npy", localy)
            
        self.save_agent(
            f"{save_dir}/actor_final.weights.h5",
            f"{save_dir}/critic_final.weights.h5"
        )
        self.save_metrics(f"{save_dir}/metrics_final.csv")
        
        return self.metrics
    
    def evaluate(self, num_episodes=10):
        evaluation_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_similarities': [],
            'paths': []
        }
        
        for episode in range(num_episodes):
            # state, x, m, localx, localy, pher_map = self.env.reset()
            state , info = self.env.reset()
            x = info['robot_position']
            m = info['map']
            localx = info['path_x']
            localy = info['path_y']
            pher_map = info['pheromone_map']

            episode_reward = 0
            episode_step = 0
            done = False
            
            while not done:
                episode_step += 1
                
                probs = self.agent.actor.predict(state)
                action = np.argmax(probs[0])

                # next_state, x, m, localx, localy, done, reward, _, pher_map = self.env.step(
                #     action, x, m, localx, localy, pher_map
                # )
                next_state, reward, done, info = self.env.step(action, x, m, localx, localy, pher_map)
                if episode_step >= self.training_params.get('max_steps', 1000) or done:
                    done = True
                    r, info= self.env.compute_final_reward()
                    sim = info['map_similarity']
                    reward += r
                episode_reward += reward
                state = next_state
            
            # Store evaluation metrics
            evaluation_metrics['episode_rewards'].append(float(episode_reward))
            evaluation_metrics['episode_lengths'].append(episode_step)
            evaluation_metrics['episode_similarities'].append(sim)
            evaluation_metrics['paths'].append({
                'localx': localx.tolist() if isinstance(localx, np.ndarray) else localx,
                'localy': localy.tolist() if isinstance(localy, np.ndarray) else localy
            })
        
        return evaluation_metrics