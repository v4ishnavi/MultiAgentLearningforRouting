from envi import Environment, Customer, Vehicle
from DQN import DQN_c2s, ReplayBuffer, DQN_vrp
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)


# Hyperparameters
C2S_STATE_DIM = 19  # From the environment
C2S_ACTION_DIM = 5  # Discrete actions
C2S_BUFFER_CAPACITY = 100000  # Replay buffer capacity
BATCH_SIZE = 512  # Mini-batch size
C2S_LEARNING_RATE = 0.001  # Learning rate for Adam optimizer
C2S_GAMMA = 0.9  # Discount factor
C2S_TARGET_UPDATE = 10  # Frequency of target network updates

VRP_STATE_DIM = 17  # From the environment
VRP_ACTION_DIM = 1  # Discrete actions
VRP_BUFFER_CAPACITY = 100000  # Replay buffer capacity
VRP_LEARNING_RATE = 0.001  # Learning rate for Adam optimizer
VRP_EPSILON_START = 1.0  # Initial exploration rate
VRP_EPSILON_DECAY = 0.999  # Decay rate of exploration rate
VRP_TARGET_UPDATE = 4  # Frequency of target network updates

EPISODES = 1000  # Number of training episodes

# Initialize models, optimizer, and replay buffer
dqn_c2s = DQN_c2s(C2S_STATE_DIM, C2S_ACTION_DIM)
# target_dqn_c2s = DQN_c2s(C2S_STATE_DIM, C2S_ACTION_DIM)
# target_dqn_c2s.load_state_dict(dqn_c2s.state_dict())  # Synchronize target DQN with the main DQN
# target_dqn_c2s.eval()  # Target DQN does not update during training

dqn_vrp = DQN_vrp(VRP_STATE_DIM, VRP_ACTION_DIM)


optimizer_c2s = optim.Adam(dqn_c2s.parameters(), lr=C2S_LEARNING_RATE)
loss_fn = nn.MSELoss()
replay_buffer_c2s = ReplayBuffer(C2S_BUFFER_CAPACITY)

optimizer_vrp = optim.Adam(dqn_vrp.parameters(), lr=C2S_LEARNING_RATE)
replay_buffer_vrp = ReplayBuffer(VRP_BUFFER_CAPACITY)

c2s_flag = 0
vrp_flag = 0

# Training loop for DQN with delayed rewards
epsilon = VRP_EPSILON_START  # Initial exploration rate
for episode in range(EPISODES):
    # Reset the environment
    # Create an environment
    vrp_episode_loss = 0
    if c2s_flag and vrp_flag:
        env = Environment(1, 1, dqn_c2s, dqn_vrp)
    elif c2s_flag:
        env = Environment(1, 0, dqn_c2s, None)
    elif vrp_flag:
        env = Environment(0, 1, None, dqn_vrp)
    else:
        env = Environment(0, 0)

    # T = 0
    env.initialize_environment()
    c2s_rewards, vrp_rewards = env.env_step(epsilon) 

    # while not done:
    #     # Epsilon-greedy policy
    #     if random.random() < epsilon:
    #         action = random.randint(0, ACTION_DIM - 1)  # Random action
    #     else:
    #         with torch.no_grad():
    #             q_values = dqn(torch.FloatTensor(state))
    #             action = torch.argmax(q_values).item()  # Greedy action

    #     # Execute action in the environment
    #     next_state = env.c2s_step(action)  # Update environment with the action
    #     states_batch.append(state)
    #     actions_batch.append(action)

    #     # Collect batch and compute delayed rewards after 200 decisions
    #     if len(states_batch) >= 200:  # Simulated delayed feedback
    #         rewards = env.compute_c2s_reward()  # Compute batch reward from your environment
    #         rewards_batch = [rewards] * len(states_batch)
    #         done = True  # End the batch

    for time_step in range(5):
        # Add experiences to replay buffer
        for i in range(len(c2s_rewards) -1):
            replay_buffer_c2s.add(c2s_rewards[i][0], c2s_rewards[i][1], c2s_rewards[i][2], c2s_rewards[i+1][0], False)
            # state, action, reward, next_state, done

        for agent in vrp_rewards:
            for vehicle in agent:
                for i in range(len(vehicle)):
                    replay_buffer_vrp.add(vehicle[i][0], vehicle[i][2])
                    # state, reward
        
        # for state, action, reward in zip(states_batch, actions_batch, rewards_batch):
        #     replay_buffer.add(state, action, reward, state, False)  # 'next_state' placeholder for now

        # Train the model if buffer has enough samples
        if len(replay_buffer_c2s) >= BATCH_SIZE:
            # Sample a batch
            states, actions, rewards, next_states, dones = replay_buffer_c2s.sample(BATCH_SIZE)

            # Convert to PyTorch tensors
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.BoolTensor(dones)

            # Compute Q-values and targets
            q_values = dqn_c2s(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
            with torch.no_grad():
                next_q_values = dqn_c2s(next_states).max(1)[0]
                targets = rewards + C2S_GAMMA * next_q_values

            # Compute loss and optimize
            loss_c2s = loss_fn(q_values, targets)
            optimizer_c2s.zero_grad()
            loss_c2s.backward()
            optimizer_c2s.step()

        if len(replay_buffer_vrp) >= BATCH_SIZE:
            # Sample a batch
            states,rewards = replay_buffer_vrp.sample(BATCH_SIZE)

            # Convert to PyTorch tensors
            states = torch.FloatTensor(states)

            # Compute Q-values and targets
            optimizer_vrp.zero_grad()
            q_values_pred = dqn_vrp(states)
            # Compute loss and optimize
            loss = loss_fn(q_values_pred, rewards)
            loss.backward()
            optimizer_vrp.step()
            vrp_episode_loss += loss.item()

    # Log progress
    print(f"Episode {episode}, c2s_Loss: {loss_c2s.item()}, vrp_Loss: {vrp_episode_loss}")
    epsilon = max(0.0, epsilon * VRP_EPSILON_DECAY)
    

print("Training completed.")
