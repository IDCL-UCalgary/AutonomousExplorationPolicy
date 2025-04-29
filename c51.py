import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Softmax
import numpy as np
import math
from collections import deque
import random


class ReplayBuffer:
    def __init__(self, capacity=4000, batch_size=6):
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size

    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self):
        sample = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, done = map(
            np.asarray, zip(*sample))
        # Reshape states based on their dimensions
        state_shape = states[0].shape
        states = np.array(states).reshape(self.batch_size, *state_shape)
        next_states = np.array(next_states).reshape(self.batch_size, *state_shape)
        return states, actions, rewards, next_states, done

    def size(self):
        return len(self.buffer)


class ActionValueModel:
    def __init__(self, state_dim, action_dim, atoms, z, lr=0.0001):
        """
        Action Value Distribution Model for C51
        
        Args:
            state_dim: Dimensions of the state space (height, width, channels)
            action_dim: Number of possible actions
            atoms: Number of atoms in the support distribution
            z: Support for the categorical distribution
            lr: Learning rate
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.atoms = atoms
        self.z = z
        self.lr = lr

        self.opt = tf.keras.optimizers.Adam(lr)
        self.criterion = tf.keras.losses.CategoricalCrossentropy()
        self.model = self.create_model()

    def create_cnn(self):
        """Create the convolutional neural network backbone"""
        return tf.keras.Sequential([
            Input((self.state_dim[0], self.state_dim[1], self.state_dim[2])),
            Conv2D(32, kernel_size=(8, 8), strides=(4, 4), activation='relu'),
            Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu'),
            Conv2D(128, kernel_size=(4, 4), strides=(2, 2), activation='relu'),
            Flatten(),
        ])
    
    def create_model(self):
        """Create the distributional Q-network"""
        cnn = self.create_cnn()
        cnn.build((None, self.state_dim[0], self.state_dim[1], self.state_dim[2]))
        y = Dense(512, activation='relu')(cnn.output)
        h2 = Dense(64, activation='relu')(y)
        # Create an output head for each action
        outputs = []
        for _ in range(self.action_dim):
            outputs.append(Dense(self.atoms, activation='softmax')(h2))
        model = tf.keras.Model(cnn.input, outputs)
        return model

    def train(self, x, y):
        """Train the model with distributional targets"""
        y = tf.stop_gradient(y)
        with tf.GradientTape() as tape:
            logits = self.model(x)
            loss = self.criterion(y, logits)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def predict(self, state):
        """Get action distribution for a state"""
        return self.model.predict(state, verbose=0)

    def get_action(self, state, ep=0):
        """
        Select an action using epsilon-greedy policy
        
        Args:
            state: Current state observation
            ep: Episode number for epsilon calculation
            
        Returns:
            Selected action
        """
        state = np.reshape(state, [1, *self.state_dim])
        eps = 1. / ((ep / 10) + 1)
        if np.random.rand() < eps:
            return np.random.randint(0, self.action_dim)
        else:
            return self.get_optimal_action(state)
    
    def get_action_test(self, state):
        """Select the best action without exploration"""
        state = np.reshape(state, [1, *self.state_dim])
        return self.get_optimal_action(state)

    def get_optimal_action(self, state):
        """Get the action with highest expected value"""
        z = self.model.predict(state, verbose=0)
        z_concat = np.vstack(z)
        # Calculate the expected value for each action
        q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1)
        return np.argmax(q)

    def save_weights(self, filepath):
        """Save the model weights"""
        self.model.save_weights(filepath)
        
    def load_weights(self, filepath):
        """Load the model weights"""
        self.model.load_weights(filepath)


class C51Agent:
    """Categorical DQN (C51) Agent implementation"""
    
    def __init__(self, state_dim, action_dim, 
                 lr=0.0001, batch_size=6, 
                 gamma=0.99, atoms=51,
                 v_min=-4.0, v_max=4.0,
                 buffer_capacity=4000):
        """
        Initialize C51 Agent
        
        Args:
            state_dim: Dimensions of the state space
            action_dim: Number of possible actions
            lr: Learning rate
            batch_size: Batch size for training
            gamma: Discount factor
            atoms: Number of atoms in the distribution
            v_min: Minimum value of the support
            v_max: Maximum value of the support
            buffer_capacity: Replay buffer capacity
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        
        # Distribution parameters
        self.atoms = atoms
        self.v_max = v_max
        self.v_min = v_min
        self.delta_z = float(self.v_max - self.v_min) / (self.atoms - 1)
        self.z = [self.v_min + i * self.delta_z for i in range(self.atoms)]
        
        # Initialize models and buffer
        self.q = ActionValueModel(self.state_dim, self.action_dim, self.atoms, self.z, lr)
        self.q_target = ActionValueModel(self.state_dim, self.action_dim, self.atoms, self.z, lr)
        self.buffer = ReplayBuffer(capacity=buffer_capacity, batch_size=batch_size)
        
        # Initialize target network with same weights
        self.target_update()

    def target_update(self):
        """Update target network with current network weights"""
        weights = self.q.model.get_weights()
        self.q_target.model.set_weights(weights)

    def get_action(self, state, ep=0):
        """Select an action using epsilon-greedy policy"""
        return self.q.get_action(state, ep)
    
    def get_action_test(self, state):
        """Select the best action without exploration"""
        return self.q.get_action_test(state)
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer"""
        self.buffer.put(state, action, reward, next_state, done)
    
    def save_models(self, filepath_prefix="./c51_model"):
        """Save the agent's models"""
        self.q.save_weights(f"{filepath_prefix}_q")
        self.q_target.save_weights(f"{filepath_prefix}_q_target")
    
    def load_models(self, filepath_prefix="./c51_model"):
        """Load the agent's models"""
        self.q.load_weights(f"{filepath_prefix}_q")
        self.q_target.load_weights(f"{filepath_prefix}_q_target")

    def train(self):
        """
        Train the agent using a batch from replay buffer
        
        Returns:
            Loss value or None if buffer is too small
        """
        if self.buffer.size() < self.batch_size:
            return None
            
        states, actions, rewards, next_states, dones = self.buffer.sample()
        
        # Get distributional predictions
        z = self.q.predict(next_states)
        z_ = self.q_target.predict(next_states)
        
        # Convert to Q-values to find best actions
        z_concat = np.vstack(z)
        q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1)
        q = q.reshape((self.batch_size, self.action_dim), order='F')
        next_actions = np.argmax(q, axis=1)
        
        # Initialize target distributions
        m_prob = [np.zeros((self.batch_size, self.atoms)) for _ in range(self.action_dim)]
        
        # Calculate projected distributional targets
        for i in range(self.batch_size):
            if dones[i]:
                # Terminal state - just return the reward
                Tz = min(self.v_max, max(self.v_min, rewards[i]))
                bj = (Tz - self.v_min) / self.delta_z
                l, u = math.floor(bj), math.ceil(bj)
                m_prob[actions[i]][i][int(l)] += (u - bj)
                m_prob[actions[i]][i][int(u)] += (bj - l)
            else:
                # Project distributional targets according to C51 algorithm
                for j in range(self.atoms):
                    Tz = min(self.v_max, max(self.v_min, rewards[i] + self.gamma * self.z[j]))
                    bj = (Tz - self.v_min) / self.delta_z
                    l, u = math.floor(bj), math.ceil(bj)
                    m_prob[actions[i]][i][int(l)] += z_[next_actions[i]][i][j] * (u - bj)
                    m_prob[actions[i]][i][int(u)] += z_[next_actions[i]][i][j] * (bj - l)
        
        # Train the model
        loss = self.q.train(states, m_prob)
        return loss