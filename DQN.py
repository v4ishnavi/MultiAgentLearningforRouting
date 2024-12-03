import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Define the DQN model
class DQN_c2s(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN_c2s, self).__init__()
        self.fc1 = nn.Linear(input_dim, 76)
        self.fc2 = nn.Linear(76, 38)
        self.fc3 = nn.Linear(38, output_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


class DQN_vrp(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN_vrp, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 8)
        self.fc5 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = self.fc5(x)
        return x

# Replay Buffer class
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.bool_))

    def __len__(self):
        return len(self.buffer)

class ReplayBuffer_vrp:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, reward):
        self.buffer.append((state, reward))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, rewards = zip(*batch)
        return (np.array(states, dtype=np.float32),
                np.array(rewards, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


