import numpy as np 
import random
import torch
from gae import GraphAutoEncoder
from torch_geometric.data import Data
import torch_geometric.utils as utils
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from maxsat import maxsat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)


class Customer():
    def __init__(self, id, arrival, T=100, SAT=False):
        self.id = id
        self.location = np.array((np.random.uniform(-100, 100), np.random.uniform(-100, 100))) #!change TO numpy array??
        self.demand = np.random.randint(1, 11)
        self.time_window = (np.random.randint(T // 5, 4 *(T //5)), 0)
        self.time_window = (self.time_window[0], self.time_window[0] + np.random.randint(T // 10, 2 * T))
        self.assignment = -1 # -1:unassigned, 0-3: assigned to warehouse, 4 deferred
        self.deferred = 0
        self.vehicle_id = -1 # -1: not assigned, >=0 assigned to vehicle
        self.arrival = arrival
        self.SAT = SAT
        
        self.cluster = None
        self.served = False

class Vehicle():
    def __init__(self, id, location, capacity, time):
        self.id = id
        self.location = location
        self.capacity = capacity
        self.current_cap = 0
        self.customers = []
        self.speed = 10

        self.available = time # time taken for the vehicle to satisfy customers in current sub-tour


class Environment():
    def __init__(self, vrp=1, c2s=1, dqn_c2s=None, dqn_vrp=None, 
    gae_model_path = "./model.pt"):
        self.vrp = vrp
        self.c2s = c2s
        self.dqn_c2s = dqn_c2s
        self.dqn_vrp = dqn_vrp
        self.orders = []
        self.tau = 1000
        self.T = 100
        self.P_0max = 300
        self.cluster_n = 5
        self.vehicle_cap = 10
        self.clock = 0
        self.kappa = 2
        self.env_time = 0
        self.service_time = 1
        self.gamma = 0.9
        self.gae_input_dim = 2 
        self.gae_output_dim = 2 
        self.gae_hidden_dim = 16
        self.gae_model_path = gae_model_path

        self.gae_model = GraphAutoEncoder(self.gae_input_dim, self.gae_hidden_dim, self.gae_output_dim)
        self.gae_model.load_state_dict(torch.load(self.gae_model_path))
        self.gae_model.eval()
        self.state = self.initialize_environment()
    
    def initialize_environment(self):
        num_customers = np.random.randint(200, 300)
        env_info = {
            "warehouses": [ #!change locations to numpy arrays? 
                {"location": np.array((50, 50)), "inventory": self.P_0max, "vehicles": []},
                {"location": np.array((50, -50)), "inventory": self.P_0max, "vehicles": []},
                {"location": np.array((-50, -50)), "inventory": self.P_0max, "vehicles": []},
                {"location": np.array((-50, 50)), "inventory": self.P_0max, "vehicles": []},
            ],
            "customers": [Customer(i, arrival=0)for i in range(num_customers)],
        }

        clocs = np.array([c.location for c in env_info['customers']])
        wlocs = np.array([w['location'] for w in env_info['warehouses']])
        # Generate edges using rho-based clustering and negative sampling
        customer_edges, _ = self._generate_edges(clocs, wlocs, n=self.cluster_n)

        # Initialize GAE embeddings for customers
        customer_features = torch.tensor(clocs, dtype=torch.float)
        # Compute embeddings using the GAE model
        with torch.no_grad():
            self.gae_temp = self.gae_model.encode(customer_features, customer_edges).cpu().numpy()
        self.gae_embeddings = {}
        for embedding, customer in zip(self.gae_temp, env_info['customers']):
            self.gae_embeddings[customer.id] = embedding    
        # self.gae_embeddings = self.gae_model.encode(customer_features, customer_edges).cpu().numpy()
        self.orders = [env_info['customers'][i].id for i in range(len(env_info['customers']))] 
        # print("ENV init")
        # print(self.orders)
        # print(num_customers)
        return env_info
    
    def _generate_edges(self, clocs, wlocs, n):
        # Perform rho-based clustering and negative sampling
        ccount = len(clocs)
        adj = np.zeros((ccount, ccount))
        # Compute distances between all customers
        for i in range(ccount):
            for j in range(ccount):
                adj[i, j] = np.linalg.norm(clocs[i] - clocs[j])
        # Compute distances from customers to the nearest warehouse
        depot_dists = np.array([min([np.linalg.norm(clocs[i] - w) for w in wlocs]) for i in range(ccount)])
        # Form clusters based on nearest neighbors to warehouses
        assigned = np.zeros(ccount, dtype=bool)
        clusters = []
        while not np.all(assigned):
            new_cluster = []
            unassigned = np.where(~assigned)[0]
            nearest = unassigned[np.argmin(depot_dists[unassigned])]
            new_cluster.append(nearest)
            assigned[nearest] = True
            temp = new_cluster.copy()
            for c in temp:
                old_len = len(new_cluster)
                unassigned = np.where(~assigned)[0]
                nearest_neighbors = unassigned[np.argsort([adj[c, i] for i in unassigned])[:n]]
                for nn in nearest_neighbors:
                    if not assigned[nn]:
                        new_cluster.append(nn)
                        assigned[nn] = True
                if len(new_cluster) == old_len:
                    break
            clusters.append(new_cluster)
        # Compute cluster diameters and determine rho
        diameters = [max([adj[i, j] for i in c for j in c if i != j]) if len(c) > 1 else 0 for c in clusters]
        rho = np.mean(diameters) / 2 if diameters else 0
        # Form adjacency matrix with nodes within `rho` of a warehouse
        final_adj = np.zeros((ccount, ccount))
        for i in range(ccount):
            for j in range(ccount):
                if np.any([np.linalg.norm(clocs[i] - w) < rho for w in wlocs]) and np.any([np.linalg.norm(clocs[j] - w) < rho for w in wlocs]):
                    final_adj[i, j] = 1
        # Add negative samples
        count_positives = np.sum(final_adj == 1)
        count_negatives = 0
        while count_negatives < count_positives:
            i_rand, j_rand = np.random.randint(0, ccount), np.random.randint(0, ccount)
            if adj[i_rand, j_rand] > rho and final_adj[i_rand, j_rand] == 0:
                final_adj[i_rand, j_rand] = -1
                count_negatives += 1
        # Convert adjacency matrix to edge indices
        edges = utils.dense_to_sparse(torch.tensor(final_adj, dtype=torch.float))[0]
        return edges, rho


    def c2s_h(self):
        order = self.orders[0]
        action = -1
        if self.state['customers'][order].assignment == -1: 
            distances = []
            for warehouse in self.state['warehouses']:
                distances.append(np.linalg.norm(warehouse['location'] - self.state['customers'][order].location)) 
            indices = np.argsort(distances)
            for i in indices:
                if self.state['warehouses'][i]['inventory'] >= self.state['customers'][order].demand: 
                    action = i # self.state['warehouses'].index(self.state['warehouses'][i])
                    break
            else:
                action = len(self.state['warehouses'])
        return action
    
    def c2s_l(self):
        # dqn returns the action to be taken
        # if random.random() < epsilon:
        #     action = random.randint(1, 6)  # Random action
        # else:
        with torch.no_grad():
            q_values = self.dqn_c2s(torch.FloatTensor(self.get_c2s_observation()).to(device))
            action = torch.argmax(q_values).item()  # Greedy action
        return action   
        
    def get_c2s_observation(self):
        # The state for the c2s agent is the 19 state table in the paper
        observation = []
        customers_id = self.orders[0]
        observation.append(self.gae_embeddings[customers_id][0])
        observation.append(self.gae_embeddings[customers_id][1])
        # compute the distance between the customer and the warehouse(not saving any time by doing this earlier)
        # append quantinty of the product at each warehouse
        # print(customers_id)
        for warehouse in self.state["warehouses"]:
            observation.append(np.linalg.norm(warehouse['location'] -self.state["customers"][customers_id].location))
            observation.append(warehouse['inventory'])
        # append demand, time window the customer
        observation.append(self.state["customers"][customers_id].demand)
        observation.append(self.state["customers"][customers_id].time_window[0])
        observation.append(self.state["customers"][customers_id].time_window[1])
        # append the time
        observation.append(self.env_time)
        # append the number of times deferred
        observation.append(self.state["customers"][customers_id].deferred)
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
        self.state['customers'][order].assignment = action
        if action == 4:
            self.state['customers'][order].deferred += 1
        else:
            self.state['warehouses'][action]['inventory'] -= self.state['customers'][order].demand
            self.state['customers'][order].vehicle_id = -1

        if self.state['customers'][order].assignment == 4:
            self.orders.append(order)
        self.orders = self.orders[1:]
        if self.orders == []:
            return None 
        return self.get_c2s_observation() #!change from internal to external class fn
    
    def compute_c2s_reward(self, optimized_tour, id):
        # organise into sub-tours
        tours = optimized_tour

        rewards = {}
        Li = {}
        Di = {}
        Ui = {}
        for vehicle_id, _ in tours.items():
            c_list = [self.state['customers'][c_id] for c_id in tours[vehicle_id]]
            # compute tour distance for L
            path = [(vehicle_id, c_id) for c_id in tours[vehicle_id]]
            L = self.compute_distance(id, path) / (len(c_list) * 282.84)

            # proportion of empty space at the beginning
            U = 1 - (sum([self.state['customers'][c_id].demand for c_id in tours[vehicle_id]]) / self.vehicle_cap)

            D, F = [], []
            time = self.env_time
            for i, c in enumerate(c_list):
                D.append(np.linalg.norm(c.location - self.state['warehouses'][id]['location']))

                if i == 0:
                    t_p = D[0] / self.state['warehouses'][id]['vehicles'][vehicle_id].speed
                    if time + t_p <= c.time_window[1]:
                        F.append(1)
                        time = max(c.time_window[0], time + t_p) + self.service_time
                    else:
                        F.append(0)
                        time = time + t_p
                else:
                    t_p = self.Euclidean_CC(c_list[i-1].id, c.id) / self.state['warehouses'][id]['vehicles'][vehicle_id].speed
                    if time + t_p >= c.time_window[0] and time + t_p <= c.time_window[1]:
                        F.append(1)
                        time = time + t_p + self.service_time
                    else:
                        F.append(0)
                        time = time + t_p

                r_c = (-1*(D[i] + L)) + F[i] - U
                if F[i] == 0: r_c -= 10
                r_c = r_c * (self.gamma**c.deferred)
                rewards[c.id] = r_c
                Li[c.id] = L
                Di[c.id] = D[i]
                Ui[c.id] = U
        
        return rewards, Li, Di, Ui



    def vrp_h(self, id, customer_ids):
        customers = [self.state['customers'][c] for c in customer_ids if self.state['customers'][c].assignment == id] #!change dict to class not

        # Sort the assigned customers at the warehouse according to their time window opening.
        customers = sorted(customers, key=lambda x: x.time_window[0])
        served = np.zeros(len(customers), dtype=bool)

        vehicle = self.state['warehouses'][id]['vehicles'][0]
        v_idx = 0

        while(np.sum(served) < len(served)):
            added = False
            for c in customers:
                if (c.demand + vehicle.current_cap <= vehicle.capacity and 
                    not served[customers.index(c)]):

                    dist = self.Euclidean_CV(c.id, id, vehicle.id)
                    time = dist / vehicle.speed

                    if vehicle.customers == [] and time + self.env_time <= c.time_window[1]:
                        added = True
                        vehicle.available = max(c.time_window[0], self.env_time + time) + self.service_time
                    
                    elif vehicle.customers != [] and time + vehicle.available <= c.time_window[1] and time + vehicle.available >= c.time_window[0]:
                        added = True
                        vehicle.available += time + self.service_time

                    if added:
                        # assign customer to vehicle
                        vehicle.customers.append(c.id)
                        vehicle.current_cap += c.demand
                        c.vehicle_id = vehicle.id
                        vehicle.location = c.location
                        served[customers.index(c)] = True
                        added = True
                        self.state['warehouses'][id]['vehicles'][v_idx] = vehicle
                        self.state['customers'][c.id].vehicle_id = vehicle.id
                        break


            if added == False and vehicle.customers == []: # no more customers to serve
                # remove vehicle
                self.state['warehouses'][id]['vehicles'].remove(vehicle)
                break
            elif added == False: # spawn new vehicle
                vehicle = Vehicle(len(self.state['warehouses'][id]['vehicles']), self.state['warehouses'][id]['location'], self.vehicle_cap, self.env_time)
                self.state['warehouses'][id]['vehicles'].append(vehicle)
                v_idx += 1

        unserved = [c.id for c in customers if not served[customers.index(c)]]

        # return tour: list of c-v assignments
        tour = []
        for vehicle in self.state['warehouses'][id]['vehicles']:
            for customer in vehicle.customers:
                tour.append((customer, vehicle.id))
        return tour, unserved

    def vrp_l(self, state):
        with torch.no_grad():
            return self.dqn_vrp(torch.tensor(state))

    def vrp_init(self, customer_ids):
        # set up one agent for each warehouse
        # need to change this to only when there is no vehicle at the warehouse
        for warehouse in self.state['warehouses']:
            warehouse['vehicles'].append(Vehicle(len(warehouse['vehicles']), warehouse['location'], self.vehicle_cap, self.env_time))

        self.cluster_info = [{}, {}, {}, {}]
        for c in customer_ids:
            cu = self.state['customers'][c]
            if cu.vehicle_id == -1 and cu.assignment != 4:
                self.state['customers'][c].arrival = self.env_time

        for i in range(4):
            clocs = [self.state['customers'][c].location for c in customer_ids if self.state['customers'][c].assignment == i and self.state['customers'][c].cluster == None]
            c_idx = [self.state['customers'][c].id for c in customer_ids if self.state['customers'][c].assignment == i and self.state['customers'][c].cluster == None] 
            cluster_indices, centroids, radii, rho = self._vrp_cluster_gen(c_idx, clocs, self.cluster_n, self.state['warehouses'][i]['location'])
            for c in c_idx:
                self.state['customers'][c].cluster = cluster_indices[c]
            self.cluster_info[i]['cluster_indices'] = cluster_indices
            self.cluster_info[i]['centroids'] = centroids
            self.cluster_info[i]['radii'] = radii
            self.cluster_info[i]['rho'] = rho


        # initialize the state for the vrp agent
        self.vrp_states = [None for _ in range(4)] # placeholder for actual calculation
        self.vrp_actions = [[] for _ in range(4)]

    def _vrp_cluster_gen(self, c_ids, clocs, n, depot):
        # generate distance matrix
        # clocs = [customer.location for customer in self.state['customers'] if customer.assignment == w + 1] # and customer.cluster == None
        ccount = len(clocs)
        adj = np.zeros((ccount, ccount))
        for i in range(ccount):
            for j in range(ccount):
                adj[i, j] = np.linalg.norm(clocs[i] - clocs[j])
        depot_dists = np.array([np.linalg.norm(clocs[i] - depot) for i in range(ccount)])

        assigned = np.zeros(ccount, dtype=bool)
        clusters = []

        while not np.all(assigned):
            new_cluster = []

            # find the closest unassigned customer to depot
            unassigned = np.where(~assigned)[0]
            nearest = unassigned[np.argmin(depot_dists[unassigned])]
            new_cluster.append(nearest)
            assigned[nearest] = True

            # add n nearest unassigned customers to the cluster till no change
            temp = new_cluster.copy()
            for c in temp:
                old_len = len(new_cluster)
                unassigned = np.where(~assigned)[0]
                nearest_neighbors = unassigned[np.argsort([adj[c, i] for i in unassigned])[:n]]
                for nn in nearest_neighbors:
                    if not assigned[nn]:
                        new_cluster.append(nn)
                        assigned[nn] = True
                if len(new_cluster) == old_len:
                    break
            
            clusters.append(new_cluster)
        # neighbourhood radius
        # find cluster diameters
        radii = [np.max([adj[i, j] for i in c for j in c])/2 for c in clusters]
        rho = np.max(radii)

        # generate cluster indices for each customer
        # cluster_indices = np.zeros(ccount, dtype=int)
        # for i, c in enumerate(clusters):
        #     for j in c:
        #         cluster_indices[j] = i
        cluster_indices = {}
        for i, c in enumerate(clusters):
            for j in c:
                c_id = c_ids[j]
                cluster_indices[c_id] = i

        # cluster centroids
        centroids = [np.mean(np.array([clocs[ci] for ci in c]), axis=0) for c in clusters]
        
        return cluster_indices, centroids, radii, rho

    def compute_feasible_actions(self, vrp_id):
        # find the list of vehicles and customers
        vehicles = self.state['warehouses'][vrp_id]['vehicles']
        customers = [customer for customer in self.state['customers'] if (customer.assignment == vrp_id and customer.vehicle_id == -1)] # customer.vehicle_id == -1 !change

        # find the feasible actions
        # check distance between each vehicle and customer
        # check if the vehicle has enough capacity
        # check if the customer has not been assigned to a vehicle
        feasible_actions = []
        for vehicle in vehicles:
            for customer in customers:
                if (vehicle.current_cap + customer.demand <= vehicle.capacity):
                    # and np.linalg.norm(np.array(vehicle.location) - np.array(customer.location)) / vehicle.speed
                    # <= customer.time_window[1] - self.clock):
                    dist = self.Euclidean_CV(customer.id, vrp_id, vehicle.id)
                    time = dist / vehicle.speed
                    if vehicle.customers == [] and time + self.env_time <= customer.time_window[1]:
                        feasible_actions.append((vehicle.id, customer.id))
                    elif vehicle.customers != [] and time + vehicle.available <= customer.time_window[1]:
                        feasible_actions.append((vehicle.id, customer.id))

        return feasible_actions

    def get_vrp_observation(self, action, id):
        vehicle_id, customer_id = action
        vehicle = self.state['warehouses'][id]['vehicles'][vehicle_id]
        customer = self.state['customers'][customer_id]

        # generate 17 length action vector

        d = self.Euclidean_CV(customer_id, id, vehicle_id)
        b_d_short = (d < self.cluster_info[id]['rho']) # is d < neighborhood radius
        t = self.Euclidean_CV(customer_id, id, vehicle_id) / vehicle.speed # time taken to travel to c

        b_t_short = (t + vehicle.available <= customer.time_window[1]) # is t < time window end
        ngb = np.linalg.norm(vehicle.location - self.cluster_info[id]['centroids'][customer.cluster]) < self.cluster_info[id]['radii'][customer.cluster] # distance from vehicle to cluster centroid

        customer_list = [customer.id for customer in self.state['customers'] if customer.assignment == id and customer.arrival == self.env_time] # filter out previous time steps

        c_left = False
        non_d = np.inf # distance from c to nearest non-member
        cust_non_d = None
        for cidx in customer_list:
            cu = self.state['customers'][cidx]
            if cu.cluster != customer.cluster:
                # non_d = min(non_d, np.linalg.norm(np.array(cu.location) - np.array(vehicle.location)))
                if ngb:
                    non_d = min(non_d, self.Euclidean_CV(cu.id, id, vehicle_id))
                if cust_non_d == None:
                    cust_non_d = cu
                else:
                    if self.Euclidean_CV(cu.id, id, vehicle_id) < self.Euclidean_CV(cust_non_d.id, id, vehicle_id):
                        cust_non_d = cu
        if non_d == np.inf:
            non_d = 0


        if not ngb:
            for cidx in customer_list:
                cu = self.state['customers'][cidx]
                if cu.cluster == customer.cluster and cu.vehicle_id == -1:
                    c_left = True
                    break
            
        drop_far = False
        drop_cls = True
        if c_left:
            for cidx in customer_list:
                cu = self.state['customers'][cidx]

                if cu.vehicle_id == -1 and not drop_far:
                    if np.linalg.norm(cu.location - 
                                      self.state['warehouses'][id]['location']) > self.Euclidean_CV(
                                      cidx, id, vehicle_id):
                        drop_far = True

                if cu.vehicle_id == -1 and drop_cls:
                    if self.Euclidean_CV(cu.id, id, vehicle_id) < self.cluster_info[id]['rho']:
                        drop_cls = False

                if drop_far and not drop_cls:
                    break

        # drop_long = do something
        # is the distance from dropped customers to nearest non-member more than distance from loc to dropped customer
        drop_long = False
        if c_left:
            for cidx in customer_list:
                cu = self.state['customers'][cidx]
                if cu.cluster == customer.cluster and cu.vehicle_id == -1:
                    if self.Euclidean_CV(cu.id, id, vehicle_id) > self.Euclidean_CV(cust_non_d.id, id, vehicle_id):
                        drop_long = True
                        break


        served = len(vehicle.customers) # what if a customer's time window was missed? are they still counted as served?

        remaining_demand = sum([self.state['customers'][cidx].demand for cidx in customer_list
                            if self.state['customers'][cidx].cluster == customer.cluster and 
                            self.state['customers'][cidx].vehicle_id == -1])
        cls_dem = (vehicle.capacity - vehicle.current_cap >= remaining_demand)


        # hops How many cluster members of c can be served before c
        # cls_tim Is every cluster member feasible following c
        hops = 0
        cls_tim = True
        for cidx in customer_list:
            cu = self.state['customers'][cidx]
            if cu.cluster == customer.cluster and cu.vehicle_id == -1:
                distance = self.Euclidean_CV(cu.id, id, vehicle_id) + self.Euclidean_CC(customer_id, cu.id)
                time = distance / vehicle.speed + self.service_time
                if customer.time_window[1] >= time + vehicle.available and customer.time_window[0] <= time + vehicle.available:
                    hops += 1

                distance = d + self.Euclidean_CC(customer_id, cu.id)
                time = distance / vehicle.speed + self.service_time
                if cu.time_window[1] >= time + vehicle.available and cu.time_window[0] <= time + vehicle.available:
                    cls_tim = False

        # urgt How close to time window closure of c is v arriving
        urgt = customer.time_window[1] - (vehicle.available + t)

        dfrac = (t + self.service_time) / (vehicle.current_cap + customer.demand/vehicle.capacity)
        remote = np.linalg.norm(customer.location -self.cluster_info[id]['centroids'][customer.cluster])/max(1,self.cluster_info[id]['radii'][customer.cluster])

        c_members_num = len([c for c in customer_list if self.state['customers'][c].cluster == customer.cluster])
        step = np.array([d/282.84, b_d_short, t/(282.84/vehicle.speed), b_t_short, ngb, non_d/282.84, c_left, drop_far, drop_cls, drop_long, served/vehicle.capacity, cls_dem, hops/c_members_num, cls_tim, urgt/(self.T/2), dfrac, remote], dtype=np.float32)
        return step

    def vrp_step(self, action, id):
        # here taking an action means assigning a customer to a vehicle
        # store vrp state before hand for rollout and the sort
        
        vehicle_id, customer_id = action
        vehicle = self.state['warehouses'][id]['vehicles'][vehicle_id]
        customer = self.state['customers'][customer_id]

        action_vector = self.get_vrp_observation(action, id)

        # update vehicle instance
        if vehicle.customers == []:
            vehicle.available = max(customer.time_window[0], self.env_time + self.Euclidean_CV(customer_id, id, vehicle_id) / vehicle.speed) + self.service_time
        else:
            vehicle.available += ((self.Euclidean_CV(customer_id, id, vehicle_id) / vehicle.speed) + self.service_time)
        vehicle.customers.append(customer.id)
        vehicle.current_cap += customer.demand
        customer.vehicle_id = vehicle_id
        vehicle.location = customer.location
        self.state['warehouses'][id]['vehicles'][vehicle_id] = vehicle
        self.state['customers'][customer_id] = customer

        # perform a state update
        self.vrp_states[id] = action_vector
        return action_vector

    def vrp_episode(self, i, epsilon):
        self.vrp_actions[i] = self.compute_feasible_actions(i)  
        if len(self.vrp_actions[i]) == 0:
            return False
        estimated_Qs = []
        
        # for v in self.state['warehouses'][i]['vehicles']:
        #     if v.customers != []:
        #         print("hiii", v.id)
        #         # print(v.id, v.customers)
        #     else:
        #         print("empty", v.id)
        import copy

        # finding the top k actions
        temp_state = self.vrp_states[i]
        temp_vehicles = copy.deepcopy(self.state['warehouses'][i]['vehicles'])
        temp_customers = copy.deepcopy(self.state['customers'])


        for action in self.vrp_actions[i]:
            # save state and other details that change during action
            vehicle_id, customer_id = action
            # temp_state = self.vrp_states[i]
            # temp_customer = self.state['customers'][action[1]]
            # temp_vehicle = self.state['warehouses'][i]['vehicles'][vehicle_id]

            new_state = self.vrp_step(action, i)
            estimated_Qs.append(self.vrp_l(new_state).cpu().numpy()[0])

            # restore the state
            self.vrp_states[i] = temp_state
            # self.state['warehouses'][i]['vehicles'][vehicle_id] = temp_vehicle
            # self.state['customers'][customer_id] = temp_customer
            self.state['warehouses'][i]['vehicles'] = temp_vehicles
            self.state['customers'] = temp_customers

        # for v in self.state['warehouses'][i]['vehicles']:
        #     if v.customers != []:
        #         print("hiii2", v.id, len(v.customers), len(self.vrp_actions[i]))
        #         # print(v.id, v.customers)
        #     else:
        #         print("empty2", v.id)

        # print(estimated_Qs)

        # pick the top k actions
        if len(estimated_Qs) < self.kappa:
            indices = np.arange(len(estimated_Qs))
            distances = [0 for _ in range(len(estimated_Qs))]
            paths = [[] for _ in range(len(estimated_Qs))]
        else:
            indices = np.argsort(estimated_Qs)[-self.kappa:]
            distances = [0 for _ in range(len(estimated_Qs))]
            paths = [[] for _ in range(len(estimated_Qs))]

        # compute distances while doing the rollout
        if random.random() < 0: # epsilon:
            if len(estimated_Qs) < self.kappa:
                index = random.randint(0, len(estimated_Qs) - 1)
            else:
                index = random.randint(0, self.kappa - 1)
            paths[index].append(self.vrp_actions[i][index])

        else:
            # store initial state before hand
            temp_init_state = self.vrp_states[i]
            temp_vehicles = self.state['warehouses'][i]['vehicles']
            temp_customers = self.state['customers']

            # peform complete rollout for each of the top k actions
            for index in indices:
                paths[index].append(self.vrp_actions[i][index])

                self.vrp_step(self.vrp_actions[i][index], i)
                new_feasible_actions = self.compute_feasible_actions(i)

                while len(new_feasible_actions) != 0:
                    estimated_Qs = []
                    for action in new_feasible_actions:
                        vehicle_id, customer_id = action
                        temp_state = self.vrp_states[i]
                        temp_customer = self.state['customers'][customer_id]
                        temp_vehicle = self.state['warehouses'][i]['vehicles'][vehicle_id]

                        new_state = self.vrp_step(action, i)
                        estimated_Qs.append(self.vrp_l(new_state).cpu().numpy()[0])
                        # print(estimated_Qs)

                        self.vrp_states[i] = temp_state
                        self.state['warehouses'][i]['vehicles'][vehicle_id] = temp_vehicle
                        self.state['customers'][customer_id] = temp_customer

                    best_q_index = np.argmax(estimated_Qs)
                    paths[index].append(new_feasible_actions[best_q_index])
                    self.vrp_step(new_feasible_actions[best_q_index], i)
                    new_feasible_actions = self.compute_feasible_actions(i)
            
            self.vrp_states[i] = temp_init_state
            self.state['warehouses'][i]['vehicles'] = temp_vehicles
            self.state['customers'] = temp_customers

            # need helper to compute the distance of the path
            # for j in range(self.kappa):
            #     distances[j] = self.compute_distance(i, paths[j])
            
            # index = np.argmin(distances)
            distances = [np.inf for _ in range(len(paths))]
            for j in range(len(paths)):
                if paths[j] != []:
                    distances[j] = self.compute_distance(i, paths[j])
            index = np.argmin(distances)

        # print(self.state['warehouses'][i]['vehicles'][paths[index][0][0]].customers)
        # print(paths[index][0][1])

        # should use optimised_sub_tour 
        # paths[index][0] is the first action i.e. (v_id, c_id) pair
        self.vrp_step(paths[index][0], i)

        # use helper to get path in format for forward sat
        # returns a list of tours for each vehicle 

        # self.get_optimized_subtour(i, paths[index][0][0])  # forward SAT

        # check if this vehicle has any feasible actions left. If not, it leaves the depot and a new one is spawned
        vehicle_fesible = self.compute_feasible_actions(i)
        feasible = False
        for action in vehicle_fesible:
            vid, cid = action
            if vid == paths[index][0][0]: feasible = True
        if feasible == False:
            v = Vehicle(len(self.state['warehouses'][i]['vehicles']), self.state['warehouses'][i]['location'], self.vehicle_cap, self.env_time)
            self.state['warehouses'][i]['vehicles'].append(v)
            
        return True

    def compute_vrp_reward(self, optimized_tour, id):
        # organise into sub-tours
        tours = optimized_tour

        # compute the reward for each sub-tour
        warehouse_trav = []
        for vehicle_id, _ in tours.items():
            # implement sub-tour
            vehicle_list = []

            # reinitialise vehicle
            self.state['warehouses'][id]['vehicles'][vehicle_id].location = self.state['warehouses'][id]['location']
            self.state['warehouses'][id]['vehicles'][vehicle_id].current_cap = 0
            self.state['warehouses'][id]['vehicles'][vehicle_id].customers = []
            self.state['warehouses'][id]['vehicles'][vehicle_id].available = self.env_time

            c_list = [self.state['customers'][c_id] for c_id in tours[vehicle_id]]

            # calculate r_term for path
            d_p, t_p = [], []
            d_max, t_max = 0, 0
            prev_loc = self.state['warehouses'][id]['location']
            for c in c_list:
                d_p.append(np.linalg.norm(prev_loc - c.location))
                t_p.append(d_p[-1] / self.state['warehouses'][id]['vehicles'][vehicle_id].speed)
                prev_loc = c.location
            # return to the warehouse
            d_p.append(np.linalg.norm(prev_loc - self.state['warehouses'][id]['location']))
            t_p.append(d_p[-1] / self.state['warehouses'][id]['vehicles'][vehicle_id].speed)
            d_max = max(d_p)
            t_max = max(t_p)
            rho = self.cluster_info[id]['rho']
            r_term = (2 * rho) - (sum(d_p)*1/len(d_p))

            # implement subtour to get (state, action, reward) tuple
            for i in range(len(c_list)):
                c = c_list[i]
                state = self.vrp_step((vehicle_id, c.id), id)
                reward = ((rho - d_p[i]) / d_max) + ((self.T - t_p[i]) / t_max) + (self.gamma**(len(c_list) - i) * r_term)
                vehicle_list.append((state, (vehicle_id, c.id), reward))
            warehouse_trav.append(vehicle_list)

        return warehouse_trav


    def env_step(self, epsilon_vrp, epsilon_c2s):
        # execute the c2s agent
        # iterate through the list of unassigned customers and use c2s to decide the assignment

        c2s_tuples = []
        # print(len(self.state['customers']))
        # print(self.env_time)
        # print(self.orders)

        # delete in beginning 
        for i in range(4):
            self.state['warehouses'][i]['vehicles'] = [] 

        for order in self.orders.copy():
            if self.orders == []:
                break
            state = self.get_c2s_observation()
            id = order
            if self.state['customers'][order].assignment == -1:
                if self.c2s == 1:
                    # epsilon greedy action selection
                    if random.random() < epsilon_c2s:
                        action = random.randint(0, 4)
                    else:
                        action = self.c2s_l()
                    # if action != 4:
                        # print("yay")
                else:
                    action = self.c2s_h()
                self.c2s_step(action)
                c2s_tuples.append((state, action, id))

        # for c in self.states.customers:
        #     if c.deferred > 0 and c.assignment != 4 and c.ve

        # execute the vrp agent
        # print('test',len([customer.id for customer in self.state['customers'] if (customer.assignment != 4)]))
        customer_list = [customer.id for customer in self.state['customers'] if (customer.vehicle_id == -1 and customer.assignment != 4)]
        # print(len(customer_list))
        self.vrp_init(customer_list)
        # print([len(self.cluster_info[i]['centroids']) for i in range(4)])

        c2s_return = []
        vrp_return = []
        for i in range(4):
            # print(len([c for c in customer_list if self.state['customers'][c].assignment == i]))
            # print(i, len(customer_list), len(self.state['customers']))
            # iterate while customers are left without vehicle assignment
            if self.vrp == 1:
                counter = 0
                while len([customer for customer in customer_list if (self.state['customers'][customer].id == i and self.state['customers'][customer].vehicle_id == -1)]) > 0:
                    value = self.vrp_episode(i, epsilon_vrp)
                    if value == False:
                        counter += 1
                    if counter == 2:
                        break
                final_tour = self.get_current_tour(i)
                # optimized_tour = self.optimise_tour(i, final_tour)
                optimized_tour = final_tour
                unserved = [c for c in customer_list if self.state['customers'][c].assignment == i and self.state['customers'][c].vehicle_id == -1]
            else:
                _, unserved = self.vrp_h(i, customer_list)
                optimized_tour = self.get_current_tour(i)

            # iterate throught the vehicles and delete empty ones
            # print(optimized_tour)
            for vehicle in self.state['warehouses'][i]['vehicles']:
                if vehicle.customers == []:
                    self.state['warehouses'][i]['vehicles'].remove(vehicle)
                else:
                    break


            # compute reward using optimized_tour
            c2s_reward,Li, Di, Ui = self.compute_c2s_reward(optimized_tour, i)
            for c in c2s_tuples:
                if self.state['customers'][c[2]].assignment == i:
                    if c[2] in unserved:
                        rew_c2s = -10 * (self.gamma**self.state['customers'][c[2]].deferred)
                        li_c2s = 0
                        di_c2s = 0
                        ui_c2s = 0
                    else:
                        rew_c2s = c2s_reward[c[2]]
                        li_c2s = Li[c[2]]
                        di_c2s = Di[c[2]]
                        ui_c2s = Ui[c[2]]

                    c2s_return.append((c[0], c[1], rew_c2s, li_c2s, di_c2s, ui_c2s))
            vrp_reward = self.compute_vrp_reward(optimized_tour, i)
            vrp_return.append(vrp_reward)

        # EPISODE DONE. UPDATE ENVIRONMENT

        # increment environment time
        self.env_time += self.T

        # set the customers who've been deferred back to unasigned
        for order in self.orders:
            if self.state['customers'][order].assignment == 4:
                self.state['customers'][order].assignment = -1
        
        # restock warehouses
        for i in range(4):
            self.state['warehouses'][i]['inventory'] = self.P_0max
        
        # generate new customers
        num_customers = np.random.randint(200, 300)
        new_customers = []
        for i in range(num_customers):
            new = Customer(len(self.state['customers']), arrival=self.env_time)
            self.state['customers'].append(new)
            new_customers.append(new.id)

        self.orders += new_customers
        # self.gae_embeddings = np.random.rand(len(self.state['customers']), 2) 

        # delete old vehicles
        # for i in range(4):
        #     self.state['warehouses'][i]['vehicles'] = [] 

        current_customers = [customer for customer in self.state['customers'] if customer.assignment == -1]
        clocs = np.array([c.location for c in current_customers])
        wlocs = np.array([w['location'] for w in self.state['warehouses']])
        customer_edges, _ = self._generate_edges(clocs, wlocs, n=5)
        
        # print(len(self.orders), len(current_customers))
        for o in self.orders:
            if self.state['customers'][o].assignment != -1:
                # print('Assigned customer', self.state['customers'][o].assignment, self.state['customers'][o].deferred, self.state['customers'][o].arrival, self.state['customers'][o].vehicle_id)
                break

        customer_features = torch.tensor(clocs, dtype=torch.float)
        with torch.no_grad():
            gae_embeddings_new = self.gae_model.encode(customer_features, customer_edges).cpu().numpy()
        # make all old embedding None
        self.gae_embeddings = {}
        for i in range(len(gae_embeddings_new)):
            self.gae_embeddings[current_customers[i].id] = gae_embeddings_new[i]

        print('Time step done')
        return c2s_return, vrp_return


    def compute_distance(self, i, path):
        tour = {}
        for action in path:
            vehicle_id, customer_id = action
            if vehicle_id not in tour:
                tour[vehicle_id] = []
            tour[vehicle_id].append(customer_id)
        d = 0
        for vehicle_id, _ in tour.items():
            c_list = tour[vehicle_id]
            d += np.linalg.norm(self.state['warehouses'][i]['location'] - self.state['customers'][c_list[0]].location)
            for j in range(1, len(c_list)):
                d += self.Euclidean_CC(c_list[j-1], c_list[j])
        return d

    def get_current_tour(self, i):
        tour = {}
        for vehicle in self.state['warehouses'][i]['vehicles']:
            for customer in vehicle.customers:
                if vehicle.id not in tour:
                    tour[vehicle.id] = []
                tour[vehicle.id].append(customer)
        return tour

    def forward_sat(self, i, v_id):
        # find dmax
        c_list = self.state['warehouses'][i]['vehicles'][v_id].customers
        dmax = np.linalg.norm(self.state['warehouses'][i]['location'] - self.state['customers'][c_list[0]].location)
        for j in range(1, len(c_list)):
            dmax = max(dmax, np.linalg.norm(self.state['customers'][c_list[j-1]].location - self.state['customers'][c_list[j]].location))
        
        # opportunistic customers: find unserved customers in same cluster, and pick 3 randomly
        unserved = [c.id for c in self.state['customers'] if c.vehicle_id == -1 and c.assignment == i and c.cluster == self.state['customers'][c_list[0]].cluster]
        added = random.sample(unserved, min(3, len(unserved)))
        if self.state['warehouses'][i]['vehicles'][v_id].current_cap + sum([self.state['customers'][c].demand for c in added]) > self.vehicle_cap:
            added = []

        # make list of locations
        customers_R = [c.location for c in self.state['customers'] if c.id in c_list]
        customers_A = [c.location for c in self.state['customers'] if c.id in added]
        depot = self.state['warehouses'][i]['location']

        # find the optimal tour
        result = optimize_route(customers_R, customers_A, depot, dmax, 1)
        if result == None:
            return
        
        tour = result[0]
        # print(tour, type(tour))

        # rearrange c accordig to the indexing in tour
        c_list = c_list + added
        new_c_list = [c_list[i] for i in tour if i >= 0]

        # update vehicle
        vehicle = self.state['warehouses'][i]['vehicles'][v_id]
        vehicle.customers = new_c_list
        vehicle.current_cap = sum([self.state['customers'][c].demand for c in new_c_list])
        vehicle.location = self.state['customers'][new_c_list[-1]].location
        # update availability
        vehicle.available = self.env_time
        for i, c in enumerate(new_c_list):
            if i == 0:
                dist = np.linalg.norm(self.state['warehouses'][i]['location'] - self.state['customers'][c].location)
                time = dist / vehicle.speed
                if time + self.env_time <= self.state['customers'][c].time_window[1]:
                    vehicle.available = min(self.state['customers'][c].time_window[0], self.env_time) + self.service_time
                else:
                    vehicle.available += time
            else:
                dist = self.Euclidean_CC(new_c_list[i-1], c)
                time = dist / vehicle.speed
                if (time + vehicle.available <= self.state['customers'][c].time_window[1] and 
                    time + vehicle.available >= self.state['customers'][c].time_window[0]):
                    vehicle.available += time + self.service_time
                else:
                    vehicle.available += time
        self.state['warehouses'][i]['vehicles'][v_id] = vehicle


    def tightening_sat(self, optimised_tour, id): # tightening sat
        tours = optimised_tour
        
        for vehicle_id, _ in tours.items():
            c_list = tours[vehicle_id]
            d_max = np.linalg.norm(self.state['warehouses'][id]['location'] - self.state['customers'][c_list[0]].location)
            for j in range(1, len(c_list)):
                d_max = max(d_max, np.linalg.norm(self.state['customers'][c_list[j-1]].location - self.state['customers'][c_list[j]].location))
            
            customers_R = [self.state['customers'][c].location for c in c_list]
            depot = self.state['warehouses'][id]['location']

            # find the optimal tour
            result = optimize_route(customers_R, [], depot, d_max, 5)
            if result == None:
                continue
            tour, _, _ = result

            # rearrange c according to the indexing in tour
            new_c_list = [c_list[i] for i in tour]
            self.state['warehouses'][id]['vehicles'][vehicle_id].customers = new_c_list


    def Euclidean_CC(self, i, j):
        customer_i = self.state['customers'][i].location
        customer_j = self.state['customers'][j].location
        return np.linalg.norm(customer_i - customer_j) #!change

    def Euclidean_CV(self, c, w, v): # customer, warehouse, vehicle
        customer = self.state['customers'][c].location
        vehicle = self.state['warehouses'][w]['vehicles'][v].location
        return np.linalg.norm(customer - vehicle)
