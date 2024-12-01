import numpy as np 

class Customer():
    def __init__(self, id, T=100):
        self.id = id
        self.location = (np.random.uniform(-100, 100), np.random.uniform(-100, 100))
        self.demand = np.random.randint(1, 11)
        self.time_window = (np.random.randint(T // 5, 4 *(T //5)), 0)
        self.time_window = (self.time_window[0], self.time_window[0] + np.random.randint(T // 10, 2 * T))
        self.assignment = 0 # 0:unassigned, 1-4: assigned to warehouse 1-4
        self.deferred = 0
        self.vehicle_id = -1 # -1: not assigned, >=0 assigned to vehicle
        
class Vehicle():
    def __init__(self, id, location, capacity):
        self.id = id
        self.location = location
        self.capacity = capacity
        self.current_cap = 0
        self.customers = []

class Environment():
    def __init__(self, vrp=1, c2s=1):
        self.vrp = vrp
        self.c2s = c2s
        self.state = self._initialize_environment()
        self.orders = []
        self.tau = 1000
        self.T = 100
        self.P_0max = 100
        self.vehicle_cap = 10
        self.clock = 0
        
    
    def _initialize_environment(self):
        num_customers = np.random.randint(200, 301)
        env_info = {
            "warehouses": [
                {"location": (50, 50), "inventory": self.P_0max, "vehicles": []},
                {"location": (50, -50), "inventory": self.P_0max, "vehicles": []},
                {"location": (-50, -50), "inventory": self.P_0max, "vehicles": []},
                {"location": (-50, 50), "inventory": self.P_0max, "vehicles": []},
            ],
            "customers": [Customer(i)for i in range(num_customers)],
        }
        for customer in env_info['customers']:
            customer['time_window'] = (customer['time_window'][0], customer['time_window'][0] + np.random.randint(self.T // 10, 2 * self.T))

        # init gae_embeddings
        self.gae_embeddings = np.random.rand(len(env_info['customers']), 2)
        self.env_info = env_info   
        self.orders = [env_info['customers'][i] for i in range(len(env_info['customers']))] 
        
        return env_info
    
    def _time_step(self):
        # need to generate new customers
        # after generating, pass the new list of customers to the gae and update embeddings
        pass
    
    def c2s_h(self):
        order = self.orders[0]
        action = 0
        if order['assignment'] == 0:
            distances = []
            for warehouse in self.state['warehouses']:
                distances.append(np.linalg.norm(np.array(warehouse['location']) - np.array(order['location'])))
            indices = np.argsort(distances)
            for i in indices:
                if self.state['warehouses'][i]['inventory'] >= order['demand']:
                    action = self.state['warehouses'].index(self.state['warehouses'][i]) + 1
                    break
            else:
                action = len(self.state['warehouses']) + 1
        return action
    
    def c2s_l(self):
        # dqn returns the action to be taken
        pass
        
    def _get_c2s_observation(self):
        # The state for the c2s agent is the 19 state table in the paper
        observation = []
        customers_id = self.orders[0]['id']
        observation.append(self.gae_embeddings[customers_id][0])
        observation.append(self.gae_embeddings[customers_id][1])
        # compute the distance between the customer and the warehouse(not saving any time by doing this earlier)
        # append quantinty of the product at each warehouse
        for warehouse in self.state["warehouses"]:
            observation.append(np.linalg.norm(np.array(warehouse['location']) - np.array(self.state["customers"][customers_id]['location'])))
            observation.append(warehouse['inventory'])
        # append demand, time window the customer
        observation.append(self.state["customers"][customers_id]['demand'])
        observation.append(self.state["customers"][customers_id]['time_window'][0])
        observation.append(self.state["customers"][customers_id]['time_window'][1])
        # append the time
        observation.append(self.clock)
        # append the number of times deferred
        observation.append(self.state["customers"][customers_id]['deferred'])
        # vehicle availability at each warehouse
        for warehouse in self.state["warehouses"]:
            observation.append(len(warehouse['vehicles']))

        observation = np.array(observation, dtype=np.float32)
        # observation = np.random.rand(19)  # Placeholder for aggregated C2S state
        return observation
        
    def c2s_step(self, action):
        # action is the warehouse to assign the customer to
        # action = 0 means defer
        order = self.orders[0]
        order['assignment'] = action
        if action == 5:
            order['deferred'] += 1
        else:
            self.state['warehouses'][action-1]['inventory'] -= order['demand']
            order['deferred'] = 0
        if order['deferred'] > 0:
            self.orders.append(order)
        else:
            self.orders = self.orders[1:]
        return self._get_c2s_observation()
    
    def vrp_h(self):
        pass
    
    def vrp_l(self):
        pass
    
    def _get_vrp_observation(self):
        pass
    
    def vrp_step(self, action):
        pass
    
    def env_step(self):
        pass