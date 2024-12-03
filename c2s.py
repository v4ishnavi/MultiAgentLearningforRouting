import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 76)
        self.fc2 = nn.Linear(76, 38)
        self.fc3 = nn.Linear(38, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.bool)
        )

    def __len__(self):
        return len(self.buffer)



import torch.optim as optim
import numpy as np

import torch.optim as optim
import numpy as np

def train_dqn(env, num_episodes=200, batch_size=512, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1):
    # Hyperparameters
    input_dim = 19  # State size (from your environment's observation space)
    output_dim = 5  # Action space size (4 warehouses + 1 defer)

    # Initialize models
    model = DQN(input_dim, output_dim)
    target_model = DQN(input_dim, output_dim)
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    replay_buffer = ReplayBuffer(100000)

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.randint(0, output_dim)
            else:
                with torch.no_grad():
                    q_values = model(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                    action = torch.argmax(q_values).item()

            # Take action in the environment
            next_state, reward, done = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # Update model if enough samples are in the replay buffer
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = target_model(next_states).max(1)[0]
                targets = rewards + gamma * next_q_values * (~dones)

                loss = nn.MSELoss()(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Decay exploration rate
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Update the target network periodically
        if episode % 10 == 0:
            target_model.load_state_dict(model.state_dict())

        # Monitor progress
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    return model

