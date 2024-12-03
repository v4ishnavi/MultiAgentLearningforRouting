import matplotlib.pyplot as plt
import numpy as np

def sum_of_rewards(episodes, c2s_rewards, vrp_rewards, save_path='sum_of_rewards.png'):
    episodes_range = range(1, episodes + 1)
    total_rewards = np.array(c2s_rewards) + np.array(vrp_rewards)

    plt.figure(figsize=(10, 6))
    plt.plot(episodes_range, total_rewards, label='Total Rewards', color='blue')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of Rewards')
    plt.title('Sum of C2S and VRP Rewards per Episode')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Sum of rewards plot saved to {save_path}")

def negative_dist_rewards(episodes, di_rewards, save_path='negative_dist_rewards.png'):
    episodes_range = range(1, episodes + 1)
    di_rewards = np.array(di_rewards)

    plt.figure(figsize=(10, 6))
    plt.plot(episodes_range, di_rewards, label='Negative Distance Rewards', color='red')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of Di Rewards')
    plt.title('Negative Distance Rewards per Episode')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Negative distance rewards plot saved to {save_path}")

def negative_trip_rewards(episodes, li_rewards, save_path='negative_trip_rewards.png'):
    episodes_range = range(1, episodes + 1)
    li_rewards = np.array(li_rewards)

    plt.figure(figsize=(10, 6))
    plt.plot(episodes_range, li_rewards, label='Negative Trip Rewards', color='green')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of Li Rewards')
    plt.title('Negative Trip Rewards per Episode')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Negative trip rewards plot saved to {save_path}")

def capacity_utilization_rewards(episodes, ui_rewards, save_path='capacity_utilization_rewards.png'):
    episodes_range = range(1, episodes + 1)
    ui_rewards = np.array(ui_rewards)

    plt.figure(figsize=(10, 6))
    plt.plot(episodes_range, ui_rewards, label='Capacity Utilization Rewards', color='purple')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of Ui Rewards')
    plt.title('Capacity Utilization Rewards per Episode')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Capacity utilization rewards plot saved to {save_path}")

def total_trips(episodes, total_trips, save_path='total_trips.png'):
    episodes_range = range(1, episodes + 1)
    total_trips = np.array(total_trips)

    plt.figure(figsize=(10, 6))
    plt.plot(episodes_range, total_trips, label='Total Trips (Vehicles)', color='orange')
    plt.xlabel('Episodes')
    plt.ylabel('Number of Vehicles')
    plt.title('Total Trips (Vehicles) per Episode')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Total trips plot saved to {save_path}")

def customers_per_trip(episodes, customers, trips, save_path='customers_per_trip.png'):
    episodes_range = range(1, episodes + 1)
    customers = np.array(customers)
    trips = np.array(trips)

    # To avoid division by zero
    avg_customers_per_trip = np.divide(customers, trips, out=np.zeros_like(customers, dtype=float), where=trips!=0)

    plt.figure(figsize=(10, 6))
    plt.plot(episodes_range, avg_customers_per_trip, label='Avg Customers per Trip', color='brown')
    plt.xlabel('Episodes')
    plt.ylabel('Average Number of Customers per Trip')
    plt.title('Average Number of Customers per Trip per Episode')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Customers per trip plot saved to {save_path}")

# Example usage:
# After training, you can call these functions as follows:

# import metrics

# metrics.sum_of_rewards(EPISODES, c2s_rew_per_episode, vrp_rew_per_episode)
# metrics.negative_dist_rewards(EPISODES, D_per_episode)
# metrics.negative_trip_rewards(EPISODES, L_per_episode)
# metrics.capacity_utilization_rewards(EPISODES, U_per_episode)
# metrics.total_trips(EPISODES, vehicles_per_episode)
# metrics.customers_per_trip(EPISODES, customers_per_vehicle_per_episode, vehicles_per_episode)
