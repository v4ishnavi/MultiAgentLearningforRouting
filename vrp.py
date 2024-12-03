import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# VRP Estimator Network
class VRPEstimator(nn.Module):
    def __init__(self):
        super(VRPEstimator, self).__init__()
        self.fc1 = nn.Linear(17, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 8)
        self.output = nn.Linear(8, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        x = self.tanh(self.fc4(x))
        x = self.output(x)
        return x

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.float32),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.bool)
        )

    def __len__(self):
        return len(self.buffer)

# Training Loop
def train_vrp_dqn(env, num_episodes=2000, gamma=0.95, epsilon=1.0, epsilon_decay=0.999, batch_size=512):
    # Initialize models
    model = VRPEstimator()
    target_model = VRPEstimator()
    target_model.load_state_dict(model.state_dict())  # Initialize target model
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    replay_buffer = ReplayBuffer(100000)

    timestep = 0

    for episode in range(num_episodes):
        state = env.reset()  # Reset environment
        total_reward = 0
        done = False

        while not done:
            # Softmax-based action selection for exploitation
            feasible_actions = env.get_feasible_actions(state)
            if np.random.rand() < epsilon:
                action = env.random_action()  # Exploration
            else:
                q_values = []
                for action in feasible_actions:
                    features = env.get_features(state, action)
                    q_value = model(torch.tensor(features, dtype=torch.float32).unsqueeze(0)).item()
                    q_values.append(q_value)

                # Apply softmax to the Q-values for action selection
                probabilities = np.exp(q_values) / np.sum(np.exp(q_values))
                action = feasible_actions[np.argmax(probabilities)]

            # Step in the environment
            next_state, reward, done = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            timestep += 1

            # Train every 4 times after 100 timesteps
            if timestep >= 100 and timestep % 4 == 0 and len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                targets = []
                q_values = []
                for i in range(batch_size):
                    features = env.get_features(states[i], actions[i])
                    q_value = model(torch.tensor(features, dtype=torch.float32).unsqueeze(0)).item()
                    q_values.append(q_value)

                    if dones[i]:
                        target = rewards[i]
                    else:
                        next_q_values = [
                            target_model(torch.tensor(env.get_features(next_states[i], a), dtype=torch.float32).unsqueeze(0)).item()
                            for a in env.get_feasible_actions(next_states[i])
                        ]
                        target = rewards[i] + gamma * max(next_q_values)
                    targets.append(target)

                q_values = torch.tensor(q_values, dtype=torch.float32)
                targets = torch.tensor(targets, dtype=torch.float32)

                loss = nn.MSELoss()(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Decay epsilon
        epsilon = max(0.0, epsilon * epsilon_decay)

        # Update target network every 100 timesteps
        if timestep % 100 == 0:
            target_model.load_state_dict(model.state_dict())

        # Logging
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")

    return model

# Example usage with environment
# Ensure the environment has the following methods:
# - `reset()` to initialize the environment
# - `step(action)` to execute the action and return (next_state, reward, done)
# - `random_action()` to return a random feasible action
# - `get_features(state, action)` to extract the 17-dimensional feature vector
# - `get_feasible_actions(state)` to return all feasible vehicle-customer pairs

# env = YourEnvironment()  # Replace with your environment implementation
# trained_model = train_vrp_dqn(env)
