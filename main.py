from envi import Environment, Customer, Vehicle
from DQN import DQN_c2s, ReplayBuffer, ReplayBuffer_vrp, DQN_vrp
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
C2S_EPSILON_START = 1.0  # Initial exploration rate
C2S_EPSILON_DECAY = 0.97  # Decay rate of exploration rate


VRP_STATE_DIM = 17  # From the environment
VRP_ACTION_DIM = 1  # Discrete actions
VRP_BUFFER_CAPACITY = 100000  # Replay buffer capacity
VRP_LEARNING_RATE = 0.001  # Learning rate for Adam optimizer
VRP_EPSILON_START = 1.0  # Initial exploration rate
VRP_EPSILON_DECAY = 0.999  # Decay rate of exploration rate
VRP_TARGET_UPDATE = 4  # Frequency of target network updates

EPISODES = 100  # Number of training episodes

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
replay_buffer_vrp = ReplayBuffer_vrp(VRP_BUFFER_CAPACITY)

c2s_flag = 1
vrp_flag = 0
c2s_rew_per_episode = []
D_per_episode = []
L_per_episode = []
U_per_episode = []
vrp_rew_per_episode = []
vehicles_per_episode = []
customers_per_vehicle_per_episode = []
# Training loop for DQN with delayed rewards
epsilon_vrp = VRP_EPSILON_START  # Initial exploration rate
epsilon_c2s = C2S_EPSILON_START  # Initial exploration rate
for episode in range(EPISODES):
    print(f"Episode {episode+1}")
    # Reset the environment
    # Create an environment
    vrp_episode_loss = 0
    c2s_episode_loss = 0
    if c2s_flag and vrp_flag:
        env = Environment(1, 1, dqn_c2s, dqn_vrp)
    elif c2s_flag:
        env = Environment(0, 1, dqn_c2s, None)
    elif vrp_flag:
        env = Environment(1, 0, None, dqn_vrp)
    else:
        env = Environment(0, 0)

    # T = 0
    d_for_ep = 0
    l_for_ep = 0
    u_for_ep = 0 
    vrprew_for_ep = 0
    c2srew_for_ep = 0 
    vehicles_for_ep = 0
    cust_per_veh_for_ep = 0 

    time_steps = 5
    for time_step in range(time_steps):
        c2s_rewards, vrp_rewards = env.env_step(epsilon_vrp, epsilon_c2s) 
        # c2s_rewards -> list of (agent) ->  (c[0], c[1], rew_c2s, li_c2s, di_c2s, ui_c2s)
        # vrp_rewards -> list of -> [[(state, (vid, cid), reward), (), ()], []] 
        # vrp rewards -> list of (agent) -> for each agent, list of vehicle routes -> 
        #              -> list of (state, (vid, cid), reward) for each cust 

        for c2s_tuple in c2s_rewards:
            _, _, rew_c2s, li_c2s, di_c2s, ui_c2s = c2s_tuple
            c2srew_for_ep += rew_c2s
            l_for_ep += li_c2s
            d_for_ep += di_c2s
            u_for_ep += ui_c2s
        
        for agent_vrp in vrp_rewards:
            for vehicle in agent_vrp: 
                for _, _, reward in vehicle:
                    vrprew_for_ep += reward
        
        # Handle number of vehicles
        num_vehicles = sum(len(warehouse['vehicles']) for warehouse in env.state['warehouses'])
        # print("to check if there are actually any customers for ")
        # print([i.id for i in env.state['warehouses'][0]['vehicles'][0].customers if len(env.state['warehouses'][0]['vehicles']) != 0])
        vehicles_for_ep += num_vehicles

        customers_per_vehicle = sum(len(vehicle.customers) for warehouse in env.state['warehouses'] for vehicle in warehouse['vehicles'])
        cust_per_veh_for_ep += customers_per_vehicle
        # ------------------------------------------------------------------

    # while not done:
    #     # Epsilon-greedy policy
    #     if random.random() < epsilon_vrp:
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

        # Add experiences to replay buffer
        for i in range(len(c2s_rewards) -1):
            replay_buffer_c2s.add(c2s_rewards[i][0], c2s_rewards[i][1], c2s_rewards[i][2], c2s_rewards[i+1][0], False)
            # state, action, reward, next_state, done

        for agent in vrp_rewards:
            for vehicle in agent:
                # print(vehicle)
                for i in range(len(vehicle)):
                    # print(vehicle[i])
                    replay_buffer_vrp.add(vehicle[i][0], vehicle[i][2])
                    # state, reward
        
        # for state, action, reward in zip(states_batch, actions_batch, rewards_batch):
        #     replay_buffer.add(state, action, reward, state, False)  # 'next_state' placeholder for now

        # Train the model if buffer has enough samples
        if len(replay_buffer_c2s) >= BATCH_SIZE and c2s_flag:
            # Sample a batch
            states, actions, rewards, next_states, dones = replay_buffer_c2s.sample(BATCH_SIZE)

            # Convert to PyTorch tensors
            states = torch.FloatTensor(states).to(device)
            actions = torch.LongTensor(actions).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            dones = torch.BoolTensor(dones).to(device)

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
            c2s_episode_loss += loss_c2s.item()

        if len(replay_buffer_vrp) >= BATCH_SIZE and vrp_flag:
            # Sample a batch
            states,rewards = replay_buffer_vrp.sample(BATCH_SIZE)

            # Convert to PyTorch tensors
            states = torch.FloatTensor(states).to(device)
            rewards = torch.FloatTensor(rewards).to(device).reshape(-1, 1)
            # print(rewards.shape)
            # print(states[0])

            # Compute Q-values and targets
            optimizer_vrp.zero_grad()
            q_values_pred = dqn_vrp(states)
            # print(q_values_pred)
            # for name, param in dqn_vrp.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.data)

            # Compute loss and optimize
            loss = loss_fn(q_values_pred, rewards)
            loss.backward()
            max_norm = 1.0
            torch.nn.utils.clip_grad_norm_(dqn_vrp.parameters(), max_norm)
            
            optimizer_vrp.step()
            vrp_episode_loss += loss.item()
    
    # scaling by number of customers in this episode 
    c2srew_for_ep = c2srew_for_ep/len(env.state['customers'])
    d_for_ep = d_for_ep/len(env.state['customers'])
    l_for_ep = l_for_ep/len(env.state['customers'])
    u_for_ep = u_for_ep/len(env.state['customers'])
    vrprew_for_ep = vrprew_for_ep/len(env.state['customers']) 

    # averaging out for time steps ..... 
    c2s_rew_per_episode.append(c2srew_for_ep/time_steps)
    D_per_episode.append(d_for_ep/time_steps)
    L_per_episode.append(l_for_ep/time_steps)
    U_per_episode.append(u_for_ep/time_steps)
    vrp_rew_per_episode.append(vrprew_for_ep/time_steps)
    vehicles_per_episode.append(vehicles_for_ep)
    customers_per_vehicle_per_episode.append(cust_per_veh_for_ep)  


    # print("stuff")
    # print("c2s rew")
    # print(c2s_rew_per_episode)
    # print("D ")
    # print(D_per_episode)
    # print("L")
    # print(L_per_episode)
    # print("U")
    # print(U_per_episode)
    # print("vrp rew")
    # print(vrp_rew_per_episode)
    # print("vehicles per ep")
    # print(vehicles_per_episode)
    # print("customers per vehicle")
    # print(customers_per_vehicle_per_episode)

    # Log progress
    print(f"Episode {episode + 1}, c2s_Loss: {c2s_episode_loss}, vrp_Loss: {vrp_episode_loss}")
    epsilon_vrp = max(0.0, epsilon_vrp * VRP_EPSILON_DECAY)
    epsilon_c2s = max(0.0, epsilon_c2s * C2S_EPSILON_DECAY)
    

print('c2s_rew_per_episode:', c2s_rew_per_episode)
print('D_per_episode:', D_per_episode)
print('L_per_episode:', L_per_episode)
print('U_per_episode:', U_per_episode)
print('vrp_rew_per_episode:', vrp_rew_per_episode)
print('vehicles_per_episode:', vehicles_per_episode)
print('customers_per_vehicle_per_episode:', customers_per_vehicle_per_episode)


print("Training completed.")
