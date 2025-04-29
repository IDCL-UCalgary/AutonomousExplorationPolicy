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
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.atoms = atoms
        self.z = z
        self.lr = lr

        self.opt = tf.keras.optimizers.Adam(lr)
        self.criterion = tf.keras.losses.CategoricalCrossentropy()
        self.model = self.create_model()

    def create_cnn(self):
        return tf.keras.Sequential([
            Input((self.state_dim[0], self.state_dim[1], self.state_dim[2])),
            Conv2D(32, kernel_size=(8, 8), strides=(4, 4), activation='relu'),
            Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu'),
            Conv2D(128, kernel_size=(4, 4), strides=(2, 2), activation='relu'),
            Flatten(),
        ])
    
    def create_model(self):
        cnn = self.create_cnn()
        cnn.build((None, self.state_dim[0], self.state_dim[1], self.state_dim[2]))
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
        return loss

    def predict(self, state):
        return self.model.predict(state, verbose=0)

    def get_action(self, state, ep=0):
        state = np.reshape(state, [1, *self.state_dim])
        eps = 1. / ((ep / 10) + 1)
        if np.random.rand() < eps:
            return np.random.randint(0, self.action_dim)
        else:
            return self.get_optimal_action(state)
    
    def get_action_test(self, state):
        state = np.reshape(state, [1, *self.state_dim])
        return self.get_optimal_action(state)

    def get_optimal_action(self, state):
        z = self.model.predict(state, verbose=0)
        z_concat = np.vstack(z)
        q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1)
        return np.argmax(q)

    def save_weights(self, filepath):
        self.model.save_weights(filepath)
        
    def load_weights(self, filepath):
        self.model.load_weights(filepath)


class C51Agent:
    
    def __init__(self, state_dim, action_dim, 
                 lr=0.0001, batch_size=6, 
                 gamma=0.99, atoms=51,
                 v_min=-4.0, v_max=4.0,
                 buffer_capacity=4000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        
        self.atoms = atoms
        self.v_max = v_max
        self.v_min = v_min
        self.delta_z = float(self.v_max - self.v_min) / (self.atoms - 1)
        self.z = [self.v_min + i * self.delta_z for i in range(self.atoms)]
        
        self.q = ActionValueModel(self.state_dim, self.action_dim, self.atoms, self.z, lr)
        self.q_target = ActionValueModel(self.state_dim, self.action_dim, self.atoms, self.z, lr)
        self.buffer = ReplayBuffer(capacity=buffer_capacity, batch_size=batch_size)
        
        self.target_update()

    def target_update(self):
        weights = self.q.model.get_weights()
        self.q_target.model.set_weights(weights)

    def get_action(self, state, ep=0):
        return self.q.get_action(state, ep)
    
    def get_action_test(self, state):
        return self.q.get_action_test(state)
    
    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.put(state, action, reward, next_state, done)
    
    def save_models(self, filepath_prefix="./c51_model"):
        self.q.save_weights(f"{filepath_prefix}_q")
        self.q_target.save_weights(f"{filepath_prefix}_q_target")
    
    def load_models(self, filepath_prefix="./c51_model"):
        self.q.load_weights(f"{filepath_prefix}_q")
        self.q_target.load_weights(f"{filepath_prefix}_q_target")

    def train(self):
        if self.buffer.size() < self.batch_size:
            return None
            
        states, actions, rewards, next_states, dones = self.buffer.sample()
        
        z = self.q.predict(next_states)
        z_ = self.q_target.predict(next_states)
        
        z_concat = np.vstack(z)
        q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1)
        q = q.reshape((self.batch_size, self.action_dim), order='F')
        next_actions = np.argmax(q, axis=1)
        
        m_prob = [np.zeros((self.batch_size, self.atoms)) for _ in range(self.action_dim)]
        
        for i in range(self.batch_size):
            if dones[i]:
                Tz = min(self.v_max, max(self.v_min, rewards[i]))
                bj = (Tz - self.v_min) / self.delta_z
                l, u = math.floor(bj), math.ceil(bj)
                m_prob[actions[i]][i][int(l)] += (u - bj)
                m_prob[actions[i]][i][int(u)] += (bj - l)
            else:
                for j in range(self.atoms):
                    Tz = min(self.v_max, max(self.v_min, rewards[i] + self.gamma * self.z[j]))
                    bj = (Tz - self.v_min) / self.delta_z
                    l, u = math.floor(bj), math.ceil(bj)
                    m_prob[actions[i]][i][int(l)] += z_[next_actions[i]][i][j] * (u - bj)
                    m_prob[actions[i]][i][int(u)] += z_[next_actions[i]][i][j] * (bj - l)
        
        loss = self.q.train(states, m_prob)
        return loss