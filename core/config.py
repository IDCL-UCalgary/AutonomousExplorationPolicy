from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Tuple, List, Union, Optional
import json
import yaml
import os


@dataclass
class NetworkConfig:
    
    network_type: str = "cnn"  # Options: "cnn", "mlp", "cnn_mlp"
    cnn_filters: List[int] = field(default_factory=lambda: [32, 64, 128])
    cnn_kernel_sizes: List[int] = field(default_factory=lambda: [8, 4, 4])
    cnn_strides: List[int] = field(default_factory=lambda: [4, 2, 2])
    
    hidden_sizes: List[int] = field(default_factory=lambda: [512, 256])
    activation: str = "relu"  # Options: "relu", "tanh", "elu"
    use_batch_norm: bool = False
    use_layer_norm: bool = False
    dropout_rate: float = 0.0


@dataclass
class PPOConfig:
    actor_lr: float = 3e-5
    critic_lr: float = 1e-5
    
    # PPO specific 
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # GAE 
    gamma: float = 0.99
    lambda_gae: float = 0.95
    
    # Training 
    epochs_per_update: int = 3
    
    # Network architecture
    actor_network: NetworkConfig = field(default_factory=NetworkConfig)
    critic_network: NetworkConfig = field(default_factory=NetworkConfig)


@dataclass
class DQNConfig:
    
    
    lr: float = 1e-4
    gamma: float = 0.99
    
    # Exploration parameters
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 100000
    
    target_update_freq: int = 1000
    tau: float = 0.001  
    
    # Double DQN
    use_double_dqn: bool = True
    
    # Dueling DQN
    use_dueling_dqn: bool = True
    
    # Network architecture
    network: NetworkConfig = field(default_factory=NetworkConfig)


@dataclass
class SACConfig:
    
    # Learning rates
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    alpha_lr: float = 1e-4
    
    # SAC specific parameters
    gamma: float = 0.99
    tau: float = 0.005
    
    # Automatic entropy tuning
    auto_entropy_tuning: bool = True
    target_entropy: Optional[float] = None
    
    # Network architecture
    actor_network: NetworkConfig = field(default_factory=NetworkConfig)
    critic_network: NetworkConfig = field(default_factory=NetworkConfig)


@dataclass
class RobotExplorationConfig:
    """Configuration for robot exploration environments."""
    
    # Map settings
    map_width: int = 50
    map_height: int = 50
    num_obstacles: int = 10
    obstacle_size: Tuple[int, int] = (4, 4)
    
    # Robot settings
    max_linear_velocity: float = 1.0
    max_angular_velocity: float = 1.0
    sensor_range: float = 12.0
    
    # Task settings
    max_steps: int = 100
    reward_scale: float = 1.0
    
    # Reward components
    reward_info_gain: float = 1.0
    reward_step_cost: float = -0.01
    reward_collision: float = -1.0
    reward_exploration_progress: float = 0.5
    
    # Pheromone settings
    use_pheromones: bool = False
    pheromone_decay_rate: float = 0.8


@dataclass
class TrainingConfig:
    
    # Algorithm selection
    algorithm: str = "ppo"  # Options: "ppo", "dqn", "sac"
    
    # Algorithm-specific configs (only one will be used)
    ppo_config: PPOConfig = field(default_factory=PPOConfig)
    dqn_config: DQNConfig = field(default_factory=DQNConfig)
    sac_config: SACConfig = field(default_factory=SACConfig)
    
    # Environment config
    env_config: RobotExplorationConfig = field(default_factory=RobotExplorationConfig)
    
    # Training parameters
    max_episodes: int = 1000
    max_steps_per_episode: int = 1000
    batch_size: int = 64
    
    # Evaluation parameters
    eval_frequency: int = 10
    eval_episodes: int = 5
    
    # Saving parameters
    save_frequency: int = 100
    log_dir: str = "./logs"
    
    # Seed for reproducibility
    seed: int = 42
    
    def get_algorithm_config(self) -> Union[PPOConfig, DQNConfig, SACConfig]:
        if self.algorithm == "ppo":
            return self.ppo_config
        elif self.algorithm == "dqn":
            return self.dqn_config
        elif self.algorithm == "sac":
            return self.sac_config
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")


class ConfigManager:

    @staticmethod
    def load_config(config_path: str) -> TrainingConfig:

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        _, ext = os.path.splitext(config_path)
        
        if ext.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif ext.lower() == '.json':
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {ext}")
        
        return ConfigManager.dict_to_config(config_dict)
    
    @staticmethod
    def save_config(config: TrainingConfig, config_path: str) -> None:
        config_dict = asdict(config)
        _, ext = os.path.splitext(config_path)
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        if ext.lower() in ['.yaml', '.yml']:
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        elif ext.lower() == '.json':
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {ext}")
    
    @staticmethod
    def dict_to_config(config_dict: Dict[str, Any]) -> TrainingConfig:
    
        if 'ppo_config' in config_dict and isinstance(config_dict['ppo_config'], dict):
            ppo_dict = config_dict['ppo_config']
            
            if 'actor_network' in ppo_dict and isinstance(ppo_dict['actor_network'], dict):
                ppo_dict['actor_network'] = NetworkConfig(**ppo_dict['actor_network'])
            
            if 'critic_network' in ppo_dict and isinstance(ppo_dict['critic_network'], dict):
                ppo_dict['critic_network'] = NetworkConfig(**ppo_dict['critic_network'])
            
            config_dict['ppo_config'] = PPOConfig(**ppo_dict)
        
        if 'dqn_config' in config_dict and isinstance(config_dict['dqn_config'], dict):
            dqn_dict = config_dict['dqn_config']
            
            if 'network' in dqn_dict and isinstance(dqn_dict['network'], dict):
                dqn_dict['network'] = NetworkConfig(**dqn_dict['network'])
            
            config_dict['dqn_config'] = DQNConfig(**dqn_dict)
        
        if 'sac_config' in config_dict and isinstance(config_dict['sac_config'], dict):
            sac_dict = config_dict['sac_config']
            
            if 'actor_network' in sac_dict and isinstance(sac_dict['actor_network'], dict):
                sac_dict['actor_network'] = NetworkConfig(**sac_dict['actor_network'])
            
            if 'critic_network' in sac_dict and isinstance(sac_dict['critic_network'], dict):
                sac_dict['critic_network'] = NetworkConfig(**sac_dict['critic_network'])
            
            config_dict['sac_config'] = SACConfig(**sac_dict)
        
        if 'env_config' in config_dict and isinstance(config_dict['env_config'], dict):
            config_dict['env_config'] = RobotExplorationConfig(**config_dict['env_config'])
        
        return TrainingConfig(**config_dict)
    
    @staticmethod
    def create_default_config() -> TrainingConfig:
        
        return TrainingConfig()
    
    @staticmethod
    def merge_configs(base_config: TrainingConfig, override_config: Dict[str, Any]) -> TrainingConfig:
        base_dict = asdict(base_config)
        
        # Update with override values
        ConfigManager._recursive_update(base_dict, override_config)
        
        # Convert back to TrainingConfig object
        return ConfigManager.dict_to_config(base_dict)
    
    @staticmethod
    def _recursive_update(base_dict: Dict[str, Any], override_dict: Dict[str, Any]) -> None:
        
        for key, value in override_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                ConfigManager._recursive_update(base_dict[key], value)
            else:
                base_dict[key] = value


def load_config_from_args(args):
    config = ConfigManager.create_default_config()
    
    # If config file is provided, load it
    if hasattr(args, 'config') and args.config:
        config = ConfigManager.load_config(args.config)
    
    override_dict = {}
    
    if hasattr(args, 'algorithm') and args.algorithm:
        override_dict['algorithm'] = args.algorithm
    
    if hasattr(args, 'max_episodes') and args.max_episodes:
        override_dict['max_episodes'] = args.max_episodes
    
    if hasattr(args, 'max_steps') and args.max_steps:
        override_dict['max_steps_per_episode'] = args.max_steps
    
    if hasattr(args, 'batch_size') and args.batch_size:
        override_dict['batch_size'] = args.batch_size
    
    if hasattr(args, 'save_freq') and args.save_freq:
        override_dict['save_frequency'] = args.save_freq
    
    if hasattr(args, 'log_dir') and args.log_dir:
        override_dict['log_dir'] = args.log_dir
    
    if hasattr(args, 'seed') and args.seed:
        override_dict['seed'] = args.seed
    
    if override_dict:
        config = ConfigManager.merge_configs(config, override_dict)
    
    return config