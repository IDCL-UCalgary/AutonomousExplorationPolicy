from typing import Dict, Any, Tuple, List, Union, Optional, Callable
import numpy as np
import os
import time
import json
from datetime import datetime
from tqdm import tqdm
from core.agent import BaseAgent
from core.environment import BaseEnvironment


class ExperienceBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.infos = []
    
    def add(self, state, action, reward, next_state, done, info=None):
        
        if len(self.states) >= self.capacity:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.dones.pop(0)
            if info is not None:
                self.infos.pop(0)
        
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        if info is not None:
            self.infos.append(info)
    
    def get_batch(self):
        
        if len(self.infos) > 0:
            return (self.states,self.actions,self.rewards,self.next_states,self.dones,self.infos)
        else:
            return (self.states,self.actions,self.rewards,self.next_states,self.dones)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.infos = []
    
    def __len__(self):
        return len(self.states)


class Logger:
    
    
    def __init__(self, log_dir: str):
        
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(log_dir, f"run_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        self.metrics_file = os.path.join(self.run_dir, "metrics.jsonl")
        self.config_file = os.path.join(self.run_dir, "config.json")
    
    def log_config(self, config: Dict[str, Any]):
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def log_metrics(self, metrics: Dict[str, Any]):
        
        metrics['timestamp'] = time.time()
        
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
    
    def get_checkpoint_path(self, name: str, step: int) -> str:
        checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        return os.path.join(checkpoint_dir, f"{name}_{step}")


class Trainer:
    def __init__(self, agent: BaseAgent, environment: BaseEnvironment, config: Dict[str, Any], env_builder: Optional[Callable[[], BaseEnvironment]] = None):
        self.agent = agent
        self.env = environment
        self.config = config
        self.env_builder = env_builder
        
        log_dir = config.get('log_dir', './logs')
        self.logger = Logger(log_dir)
        
        self.logger.log_config(config)
        
        self.max_episodes = config.get('max_episodes', 1000)
        self.max_steps_per_episode = config.get('max_steps_per_episode', 1000)
        self.eval_frequency = config.get('eval_frequency', 10)
        self.eval_episodes = config.get('eval_episodes', 5)
        self.save_frequency = config.get('save_frequency', 100)
        
        batch_size = config.get('batch_size', 64)
        self.buffer = ExperienceBuffer(batch_size)
    
    def train(self):
        
        total_steps = 0
        best_eval_reward = float('-inf')
        
        progress_bar = tqdm(range(self.max_episodes))
        
        for episode in progress_bar:

            state = self.env.reset()
            if isinstance(state, tuple) and len(state) == 2:
                state, info = state
            else:
                info = {}
            
            episode_reward = 0
            episode_steps = 0
            episode_start_time = time.time()
            
            self.buffer.clear()
            
            for step in range(self.max_steps_per_episode):
                action = self.agent.act(state)
                
                next_state, reward, done, info = self.env.step(action)
                
                self.buffer.add(state, action, reward, next_state, done, info)
                
                episode_reward += reward
                episode_steps += 1
                total_steps += 1
                
                if len(self.buffer) >= self.config.get('batch_size', 64) or done:
                    train_metrics = self.agent.train(self.buffer.get_batch())
                    self.buffer.clear()
                
                state = next_state
                
                if done:
                    break
            
            episode_time = time.time() - episode_start_time
            episode_metrics = {
                'episode': episode,
                'total_steps': total_steps,
                'episode_reward': episode_reward,
                'episode_steps': episode_steps,
                'episode_time': episode_time,
                'steps_per_second': episode_steps / episode_time
            }
            
            if hasattr(self.env, 'get_exploration_metrics'):
                exploration_metrics = self.env.get_exploration_metrics()
                episode_metrics.update(exploration_metrics)
            
            self.logger.log_metrics(episode_metrics)
            
            progress_bar.set_description(
                f"Episode {episode} | Reward: {episode_reward:.2f} | Steps: {episode_steps}"
            )
            
            if episode % self.eval_frequency == 0:
                eval_metrics = self.evaluate(self.eval_episodes)
                eval_reward = eval_metrics['mean_reward']
                
                self.logger.log_metrics({
                    'episode': episode,
                    'evaluation': eval_metrics
                })
                
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    self.agent.save(self.logger.get_checkpoint_path('best', episode))
            
            if episode % self.save_frequency == 0:
                self.agent.save(self.logger.get_checkpoint_path('checkpoint', episode))
        
        self.agent.save(self.logger.get_checkpoint_path('final', self.max_episodes))
        
        return {
            'total_steps': total_steps,
            'best_eval_reward': best_eval_reward
        }
    
    def evaluate(self, num_episodes: int = 5) -> Dict[str, float]:
        if self.env_builder is not None:
            eval_env = self.env_builder()
        else:
            eval_env = self.env
        
        self.agent.eval_mode()
        
        rewards = []
        steps = []
        
        for _ in range(num_episodes):
            state = eval_env.reset()
            if isinstance(state, tuple) and len(state) == 2:
                state, _ = state
            
            episode_reward = 0
            episode_steps = 0
            done = False
            
            while not done and episode_steps < self.max_steps_per_episode:
                action = self.agent.act(state, deterministic=True)
                next_state, reward, done, _ = eval_env.step(action)
                episode_reward += reward
                episode_steps += 1
                state = next_state
            
            rewards.append(episode_reward)
            steps.append(episode_steps)
        
        self.agent.train_mode()
        
        eval_metrics = {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'mean_steps': np.mean(steps)
        }
        
        return eval_metrics