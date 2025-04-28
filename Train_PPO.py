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
    """A buffer to store experiences for batch training"""
    
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.clear()
    
    def clear(self):
        """Clear the buffer"""
        self.state_batch = []
        self.action_batch = []
        self.reward_batch = []
        self.next_state_batch = []
        self.done_batch = []
    
    def add(self, state, action, reward, next_state, done):
        """Add an experience to the buffer"""
        self.state_batch.append(state)
        self.action_batch.append(action)
        self.reward_batch.append(reward)
        self.next_state_batch.append(next_state)
        self.done_batch.append(done)
    
    def is_full(self):
        """Check if the buffer is full"""
        return len(self.state_batch) >= self.batch_size
    
    def get_batch(self):
        """Get the current batch of experiences"""
        return (
            self.state_batch,
            self.action_batch,
            self.reward_batch,
            self.next_state_batch,
            self.done_batch
        )


class Trainer:
    """Class to handle the training process of an agent in an environment"""
    
    def __init__(self, env_creator, agent_params, training_params):
        """
        Initialize the trainer
        
        Args:
            env_creator: Function that creates and returns an environment
            agent_params: Parameters for the PPO agent
            training_params: Parameters for the training process
        """
        self.env_creator = env_creator
        self.agent_params = agent_params
        self.training_params = training_params
        
        # Create the environment to get state and action dimensions
        self.env = self.env_creator()
        
        # Create the agent
        self.agent = PPOAgent(
            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim,
            **agent_params
        )
        
        # Create the experience buffer
        self.buffer = ExperienceBuffer(self.agent.batch_size)
        
        # Metrics storage
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
        """Save the training metrics to a CSV file"""
        df = pd.DataFrame(self.metrics)
        df.to_csv(filename, index=False)
    
    def save_agent(self, actor_path, critic_path):
        """Save the agent's models"""
        self.agent.save_models(actor_path, critic_path)
    
    def load_agent(self, actor_path, critic_path):
        """Load the agent's models"""
        self.agent.load_models(actor_path, critic_path)
    
    def train(self):
        """Run the training process"""
        max_episodes = self.training_params.get('max_episodes', 1000)
        max_steps = self.training_params.get('max_steps', 1000)
        save_freq = self.training_params.get('save_freq', 50)
        save_dir = self.training_params.get('save_dir', './models')
        save_maps = self.training_params.get('save_maps', True)
        maps_dir = self.training_params.get('maps_dir', './maps')
        
        # Make sure the directories exist
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(maps_dir, exist_ok=True)
        
        # Progress bar for episodes
        progress_bar = tqdm(range(max_episodes))
        
        for episode in progress_bar:
            # Reset the environment
            state, x, m, localx, localy, pher_map = self.env.reset()
            
            # Initialize episode variables
            episode_reward = 0
            episode_step = 0
            episode_info_gain = 0
            episode_step_cost = 0
            episode_pher_value = 0
            action_history = []
            done = False
            
            # Experience buffer for this episode
            self.buffer.clear()
            
            while not done:
                episode_step += 1
                
                # Select action according to the policy
                action = self.agent.get_action(state)
                action_history.append(action)
                
                # Take a step in the environment
                next_state, x, m, localx, localy, done, reward, plotinfo, pher_map = self.env.step(
                    action, x, m, localx, localy, pher_map
                )
                
                # Check if we reached max steps
                if episode_step >= max_steps or done:
                    done = True
                    r, totallen, sim = self.env.finish_reward(m, localx, localy, episode_step)
                    reward += r
                
                # Add experience to buffer
                state_reshaped = np.reshape(state, [1, self.env.state_dim[0], self.env.state_dim[1], self.env.state_dim[2]])
                next_state_reshaped = np.reshape(next_state, [1, self.env.state_dim[0], self.env.state_dim[1], self.env.state_dim[2]])
                self.buffer.add(state_reshaped, action, reward, next_state_reshaped, done)
                
                # Update episode metrics
                episode_reward += reward
                episode_info_gain += plotinfo[0]
                episode_step_cost += plotinfo[1]
                episode_pher_value += plotinfo[2]
                
                # Train if buffer is full or episode is done
                if self.buffer.is_full() or done:
                    train_metrics = self.agent.train(*self.buffer.get_batch())
                    self.buffer.clear()
                
                # Update state
                state = next_state
            
            # Update episode metrics
            self.metrics['episode_rewards'].append(float(episode_reward))
            self.metrics['episode_lengths'].append(episode_step)
            self.metrics['episode_info_gain'].append(episode_info_gain)
            self.metrics['episode_step_costs'].append(episode_step_cost)
            self.metrics['episode_costs'].append(totallen)
            self.metrics['episode_similarities'].append(sim)
            self.metrics['action_means'].append(np.mean(action_history))
            self.metrics['action_stds'].append(np.std(action_history))
            
            # Update progress bar description
            progress_bar.set_description(
                f"Episode {episode} | Reward: {episode_reward:.2f} | Steps: {episode_step} | Similarity: {sim:.2f}"
            )
            
            # Save checkpoint periodically
            if episode % save_freq == 0:
                # Save agent models
                self.save_agent(
                    f"{save_dir}/actor_episode_{episode}.weights.h5",
                    f"{save_dir}/critic_episode_{episode}.weights.h5"
                )
                
                # Save metrics
                self.save_metrics(f"{save_dir}/metrics_episode_{episode}.csv")
                
                # Save map data if enabled
                if save_maps:
                    np.save(f"{maps_dir}/map_episode_{episode}.npy", m)
                    np.save(f"{maps_dir}/localx_episode_{episode}.npy", localx)
                    np.save(f"{maps_dir}/localy_episode_{episode}.npy", localy)
            
        # Save final models and metrics
        self.save_agent(
            f"{save_dir}/actor_final.weights.h5",
            f"{save_dir}/critic_final.weights.h5"
        )
        self.save_metrics(f"{save_dir}/metrics_final.csv")
        
        return self.metrics
    
    def evaluate(self, num_episodes=10):
        """
        Evaluate the agent's performance
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        evaluation_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_similarities': [],
            'paths': []
        }
        
        for episode in range(num_episodes):
            # Reset the environment
            state, x, m, localx, localy, pher_map = self.env.reset()
            
            # Initialize episode variables
            episode_reward = 0
            episode_step = 0
            done = False
            
            while not done:
                episode_step += 1
                
                # Select action according to the policy (deterministically for evaluation)
                probs = self.agent.actor.predict(state)
                action = np.argmax(probs[0])
                
                # Take a step in the environment
                next_state, x, m, localx, localy, done, reward, _, pher_map = self.env.step(
                    action, x, m, localx, localy, pher_map
                )
                
                # Check if we reached max steps
                if episode_step >= self.training_params.get('max_steps', 1000) or done:
                    done = True
                    r, totallen, sim = self.env.finish_reward(m, localx, localy, episode_step)
                    reward += r
                
                # Update episode metrics
                episode_reward += reward
                
                # Update state
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


if __name__ == "__main__":
    # Example usage
    parser = argparse.ArgumentParser(description='Train a PPO agent on an exploration environment')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--steps', type=int, default=100, help='Maximum steps per episode')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--save_freq', type=int, default=50, help='Frequency of saving checkpoints')
    parser.add_argument('--save_dir', type=str, default='./models', help='Directory to save models')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    
    args = parser.parse_args()
    
    # Set device
    if not args.gpu:
        tf.config.set_visible_devices([], 'GPU')
    
    # Import environment (you'll need to uncomment and adapt based on your environment)
    # from environment import env
    # from map_generator import Map_Generator as Generator
    
    # Define environment creator function
    # def create_env():
    #     map_generator = Generator(np.random.randint(0, 100000))
    #     return env(map_generator.ref_map(), action_setting=1, pher_condition=False)
    
    # Define agent parameters
    # agent_params = {
    #     'actor_lr': 1e-4,
    #     'critic_lr': 5e-4,
    #     'entropy_beta': 0.01,
    #     'clip_ratio': 0.2,
    #     'gamma': 0.99,
    #     'lmbda': 0.95,
    #     'batch_size': args.batch_size,
    #     'n_epochs': 3
    # }
    
    # Define training parameters
    # training_params = {
    #     'max_episodes': args.episodes,
    #     'max_steps': args.steps,
    #     'save_freq': args.save_freq,
    #     'save_dir': args.save_dir,
    #     'save_maps': True,
    #     'maps_dir': './maps'
    # }
    
    # Create and run the trainer
    # trainer = Trainer(create_env, agent_params, training_params)
    # metrics = trainer.train()
    
    print("Note: Uncomment the environment import and trainer creation code to run training.")