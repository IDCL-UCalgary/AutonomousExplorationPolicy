import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten
# from tensorflow.keras.layers import InputLayer as Input

import numpy as np

class Actor:
    def __init__(self, state_dim, action_dim, actor_lr=1e-4, entropy_beta=0.01, clip_ratio=0.2):
        """
        Actor network for PPO algorithm
        
        Args:
            state_dim: Dimensions of the state space (height, width, channels)
            action_dim: Dimension of the action space (number of possible actions)
            actor_lr: Learning rate for the actor network
            entropy_beta: Coefficient for entropy bonus
            clip_ratio: PPO clipping parameter
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.entropy_beta = entropy_beta
        self.actor_lr = actor_lr 
        self.clip_ratio = clip_ratio
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(self.actor_lr, epsilon=1e-03)
    
    def create_cnn(self):
        """Create the convolutional neural network backbone"""
        cnn = tf.keras.Sequential([
            Conv2D(32, kernel_size=(8,8), strides=(4,4), activation='relu', 
                input_shape=(self.state_dim[0], self.state_dim[1], self.state_dim[2])),
            Conv2D(64, kernel_size=(4,4), strides=(2,2), activation='relu'),
            Conv2D(128, kernel_size=(4,4), strides=(2,2), activation='relu'),
            Flatten()
        ])
        # Build the model to define the output
        cnn.build(input_shape=(None, self.state_dim[0], self.state_dim[1], self.state_dim[2]))
        return cnn
    
    def create_model(self):
        """Create the actor model"""
        cnn = self.create_cnn()
        inputs = Input(shape=(self.state_dim[0], self.state_dim[1], self.state_dim[2]))
        x = cnn(inputs)
        x = Dense(512, activation='relu')(x)
        outputs = Dense(self.action_dim, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def compute_loss(self, old_policy, new_policy, actions, gaes):
        """Compute the PPO actor loss"""
        gaes = tf.stop_gradient(gaes)
        old_log_p = tf.math.log(tf.reduce_sum(old_policy * actions))
        old_log_p = tf.stop_gradient(old_log_p)
        log_p = tf.math.log(tf.reduce_sum(new_policy * actions))
        ratio = tf.math.exp(log_p - old_log_p)
        clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        surrogate = -tf.minimum(ratio * gaes, clipped_ratio * gaes)
        policy_loss = tf.reduce_mean(surrogate)
        entropy_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        entropy = entropy_loss(new_policy, new_policy)
        total_loss = policy_loss - self.entropy_beta * entropy 
        return total_loss, policy_loss, entropy

    def train(self, old_policy, states, actions, gaes):
        """Train the actor network"""
        actions = tf.one_hot(actions, self.action_dim)
        actions = tf.reshape(actions, [-1, self.action_dim])
        actions = tf.cast(actions, tf.float64)

        with tf.device('/cpu:0'):
            with tf.GradientTape() as tape:
                state_res = np.reshape(states, [len(states), self.state_dim[0], self.state_dim[1], self.state_dim[2]])
                logits = self.model([state_res], training=True)
                loss, policy_loss, entropy = self.compute_loss(old_policy, logits, actions, gaes)
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, policy_loss, entropy

    def predict(self, state):
        """Get action probabilities for a state"""
        state = np.reshape(state, [1, self.state_dim[0], self.state_dim[1], self.state_dim[2]])
        return self.model.predict(state, verbose=None)

    def save_weights(self, filepath):
        """Save the actor network weights"""
        self.model.save_weights(filepath)
        
    def load_weights(self, filepath):
        """Load the actor network weights"""
        self.model.load_weights(filepath)


class Critic:
    def __init__(self, state_dim, critic_lr=5e-4):
        """
        Critic network for PPO algorithm
        
        Args:
            state_dim: Dimensions of the state space (height, width, channels)
            critic_lr: Learning rate for the critic network
        """
        self.state_dim = state_dim
        self.model = self.create_model()
        self.critic_lr = critic_lr
        self.opt = tf.keras.optimizers.Adam(self.critic_lr, epsilon=1e-03) 
    
    def create_cnn(self):
        """Create the convolutional neural network backbone"""
        cnn = tf.keras.Sequential([
            Conv2D(32, kernel_size=(8,8), strides=(4,4), activation='relu', 
                input_shape=(self.state_dim[0], self.state_dim[1], self.state_dim[2])),
            Conv2D(64, kernel_size=(4,4), strides=(2,2), activation='relu'),
            Conv2D(128, kernel_size=(4,4), strides=(2,2), activation='relu'),
            Flatten()
        ])
        # Build the model to define the output
        cnn.build(input_shape=(None, self.state_dim[0], self.state_dim[1], self.state_dim[2]))
        return cnn
    
    def create_model(self):
        """Create the actor model"""
        cnn = self.create_cnn()
        inputs = Input(shape=(self.state_dim[0], self.state_dim[1], self.state_dim[2]))
        x = cnn(inputs)
        x = Dense(512, activation='relu')(x)
        outputs = Dense(1, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model
    

    def compute_loss(self, v_pred, td_targets):
        """Compute the critic loss"""
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    def train(self, states, td_targets):
        """Train the critic network"""
        with tf.device('/cpu:0'):
            with tf.GradientTape() as tape:
                v_pred = self.model([np.reshape(states, [len(states), self.state_dim[0], self.state_dim[1], self.state_dim[2]])], training=True)
                assert v_pred.shape == td_targets.shape
                loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss
    
    def predict(self, state):
        """Get value prediction for a state"""
        state = np.reshape(state, [1, self.state_dim[0], self.state_dim[1], self.state_dim[2]])
        return self.model.predict(state, verbose=None)
    
    def save_weights(self, filepath):
        """Save the critic network weights"""
        self.model.save_weights(filepath)
        
    def load_weights(self, filepath):
        """Load the critic network weights"""
        self.model.load_weights(filepath)


class PPOAgent:
    def __init__(self, state_dim, action_dim, 
                 actor_lr=1e-4, critic_lr=5e-4, 
                 entropy_beta=0.01, clip_ratio=0.2,
                 gamma=0.99, lmbda=0.95, 
                 batch_size=64, n_epochs=10):
        """
        PPO Agent implementation
        
        Args:
            state_dim: Dimensions of the state space
            action_dim: Dimension of the action space
            actor_lr: Learning rate for the actor network
            critic_lr: Learning rate for the critic network
            entropy_beta: Coefficient for entropy bonus
            clip_ratio: PPO clipping parameter
            gamma: Discount factor
            lmbda: GAE lambda parameter
            batch_size: Batch size for training
            n_epochs: Number of epochs to train on each batch
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lmbda = lmbda
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        
        # Initialize actor and critic networks
        self.actor = Actor(self.state_dim, self.action_dim, actor_lr, entropy_beta, clip_ratio)
        self.critic = Critic(self.state_dim, critic_lr)
    
    def get_action(self, state):
        """
        Select an action based on the current policy
        
        Args:
            state: Current state observation
            
        Returns:
            Selected action
        """
        probs = self.actor.predict(state)
        action = np.random.choice(self.action_dim, p=probs[0])
        return action
    
    def gae_target(self, rewards, v_values, next_v_value, done):
        """
        Compute GAE (Generalized Advantage Estimation) targets
        
        Args:
            rewards: List of rewards
            v_values: Value predictions for the states
            next_v_value: Value prediction for the next state
            done: Whether the episode is done
            
        Returns:
            gae: GAE values
            n_step_targets: Value targets
        """
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
        """Convert a list of arrays to a batch"""
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch
    
    def train(self, states, actions, rewards, next_states, dones):
        """
        Train the PPO agent on a batch of experiences
        
        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'actor_loss': [],
            'critic_loss': [],
            'policy_loss': [],
            'entropy': []
        }
        
        # Convert inputs to numpy arrays if they aren't already
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        # Get old action probabilities and value predictions
        old_probs = []
        v_values = []
        
        for i in range(len(states)):
            old_probs.append(self.actor.predict(states[i]))
            v_values.append(self.critic.predict(states[i])[0])
        
        old_probs = np.vstack(old_probs)
        v_values = np.array(v_values)
        
        # Get next state value prediction for the last state
        next_v_value = self.critic.predict(next_states[-1])[0] if len(next_states) > 0 else 0
        
        # Compute GAE targets
        gaes, td_targets = self.gae_target(rewards, v_values, next_v_value, dones[-1])
        td_targets = np.reshape(td_targets, (-1, 1))
        
        # Train for n_epochs
        for epoch in range(self.n_epochs):
            actor_loss, policy_loss, entropy = self.actor.train(old_probs, states, actions, gaes)
            critic_loss = self.critic.train(states, td_targets)
            
            metrics['actor_loss'].append(float(actor_loss))
            metrics['policy_loss'].append(float(policy_loss))
            metrics['entropy'].append(float(entropy))
            metrics['critic_loss'].append(float(critic_loss))
        
        return metrics
    
    def save_models(self, actor_path, critic_path):
        """Save the agent's models"""
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)
    
    def load_models(self, actor_path, critic_path):
        """Load the agent's models"""
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)