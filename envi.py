import numpy as np 
import random
import torch
from gae import GraphAutoEncoder

class Customer():
    def __init__(self, id, arrival, T=100, SAT=False):
        self.id = id
        self.location = (np.random.uniform(-100, 100), np.random.uniform(-100, 100)) #!change TO numpy array??
        self.demand = np.random.randint(1, 11)
        self.time_window = (np.random.randint(T // 5, 4 *(T //5)), 0)
        self.time_window = (self.time_window[0], self.time_window[0] + np.random.randint(T // 10, 2 * T))
        self.assignment = 0 # 0:unassigned, 1-4: assigned to warehouse 1-4
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
        self.speed = 2

        self.available = time # time taken for the vehicle to satisfy customers in current sub-tour


class Environment():
    def __init__(self, vrp=1, c2s=1, dqn_c2s=None, dqn_vrp=None, 
    gae_model_path = "model.pt"):
        self.vrp = vrp
        self.c2s = c2s
        self.dqn_c2s = dqn_c2s
        self.dqn_vrp = dqn_vrp
        self.state = self.initialize_environment() #!change
        self.orders = []
        self.tau = 1000
        self.T = 100
        self.P_0max = 100
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

        self.gae_model = GraphAutoEncoder(self.gae_input_dim, self.gae_output_dim, self.gae_hidden_dim)
        self.gae_model.load_state_dict(torch.load(self.gae_model_path))
        self.gae_model.eval()
    
    def initialize_environment(self):
        num_customers = np.random.randint(200, 300)
        env_info = {
            "warehouses": [ #!change locations to numpy arrays? 
                {"location": (50, 50), "inventory": self.P_0max, "vehicles": []},
                {"location": (50, -50), "inventory": self.P_0max, "vehicles": []},
                {"location": (-50, -50), "inventory": self.P_0max, "vehicles": []},
                {"location": (-50, 50), "inventory": self.P_0max, "vehicles": []},
            ],
            "customers": [Customer(i, arrival=0)for i in range(num_customers)],
        }

        clocs = np.array([c.location for c in env_info['customers']])
        wlocs = np.array([w['location'] for w in env_info['warehouses']])
        # Generate edges using rho-based clustering and negative sampling
        customer_edges, rho = self._generate_edges(clocs, wlocs, n=5)

        # Initialize GAE embeddings for customers
        customer_features = torch.tensor(clocs, dtype=torch.float)
        # Compute embeddings using the GAE model
        with torch.no_grad():
            self.gae_embeddings = self.gae_model.encode(customer_features, customer_edges).numpy()
        self.env_info = env_info   
        self.orders = [env_info['customers'][i] for i in range(len(env_info['customers']))] 
        
        return env_info
    
    def _generate_edges(self, clocs, wlocs, n=5):
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
        action = 0
        if order.assignment == 0: #!change 
            distances = []
            for warehouse in self.state['warehouses']:
                distances.append(np.linalg.norm(np.array(warehouse['location']) - np.array(order.location))) #!change
            indices = np.argsort(distances)
            for i in indices:
                if self.state['warehouses'][i]['inventory'] >= order.demand: #!change
                    action = self.state['warehouses'].index(self.state['warehouses'][i]) + 1
                    break
            else:
                action = len(self.state['warehouses']) + 1
        return action
    
    def c2s_l(self, epsilon):
        # dqn returns the action to be taken
        if random.random() < epsilon:
            action = random.randint(1, 6)  # Random action
        else:
            with torch.no_grad():
                q_values = self.dqn_c2s(torch.FloatTensor(self.get_c2s_observation()))
                action = torch.argmax(q_values).item()  # Greedy action
        return action
        
    def get_c2s_observation(self):
        # The state for the c2s agent is the 19 state table in the paper
        observation = []
        customers_id = self.orders[0].id #!change
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
        order.assignment = action #!change
        if action == 5:
            order.deferred += 1 #!change
        else:
            self.state['warehouses'][action-1]['inventory'] -= order.demand #!change
            order.deferred = 0 #!change
        if order.deferred > 0: #!change
            self.orders.append(order)
        self.orders = self.orders[1:]
        return self.get_c2s_observation() #!change from internal to external class fn
    
    def compute_c2s_reward(self, optimized_tour, id):
        # organise into sub-tours
        tours = {}
        for action in optimized_tour:
            vehicle_id, customer_id = action
            if vehicle_id not in tours:
                tours[vehicle_id] = []
            tours[vehicle_id].append(customer_id)

        rewards = {}
        for vehicle_id, _ in tours.items():
            c_list = [self.state['customers'][c_id] for c_id in tours[vehicle_id]]
            # compute tour distance for L
            path = [(vehicle_id, c_id) for c_id in tours[vehicle_id]]
            L = self.compute_distance(id, path) / (len(c_list) * 282.84)

            # proportion of empty space at the beginning
            U = 1 - sum([self.state['customers'][c_id].demand for c_id in tours[vehicle_id]]) / self.vehicle_cap

            D, F = [], []
            time = self.env_time
            for i, c in enumerate(c_list):
                D.append(np.linalg.norm(np.array(c.location) - np.array(self.state['warehouses'][id]['location'])))

                if i == 0:
                    t_p = D[0] / self.state['warehouses'][id]['vehicles'][vehicle_id].speed
                else:
                    t_p = self.Euclidean_CC(c_list[i-1].id, c.id) / self.state['warehouses'][id]['vehicles'][vehicle_id].speed

                if time + t_p <= c.time_window[1]:
                    F.append(1)
                    if i == 0:
                        time = min(c.time_window[0], time + t_p) + self.service_time
                    else:
                        time = time + t_p + self.service_time
                else:
                    F.append(0)
                    time = time + t_p
            
                r_c = (-1*(D[i] + L)) + F[i] - U
                if F[i] == 0: r_c -= 10
                r_c = r_c * (self.gamma**c.deferred)
                rewards[c.id] = r_c

        return rewards



    def vrp_h(self, id, customers):
        customers = [customer for customer in customers if customer.assignment == id + 1] #!change dict to class not

        # Sort the assigned customers at the warehouse according to their time window opening.
        customers = sorted(customers, key=lambda x: x.time_window[0])
        served = np.zeros(len(customers), dtype=bool)

        vehicle = self.state['warehouses'][id]['vehicles'][0]

        while(np.sum(served) < len(served)):
            added = False
            for c in customers:
                if (c.demand + vehicle.current_cap <= vehicle.capacity and 
                    not served[customers.index(c)]):

                    dist = self.Euclidean_CV(c.id, id, vehicle.id)
                    time = dist / vehicle.speed

                    if vehicle.customers == [] and time + self.env_time <= c.time_window[1]:
                        added = True
                        vehicle.available = min(c.time_window[0], self.env_time) + self.service_time
                    
                    elif vehicle.customers != [] and time + vehicle.time <= c.time_window[1]:
                        added = True
                        vehicle.available += time + self.service_time

                    if added:
                        # assign customer to vehicle
                        vehicle.customers.append(c)
                        vehicle.current_cap += c.demand
                        c.vehicle_id = vehicle.id
                        vehicle.location = c.location
                        served[customers.index(c)] = True
                        added = True
                        break


            if added == False and vehicle.customers == []: # no more customers to serve
                break
            elif added == False: # spawn new vehicle
                vehicle = Vehicle(len(self.state['warehouses'][id]['vehicles']), self.state['warehouses'][id]['location'], self.vehicle_cap, self.env_time)
                self.state['warehouses'][id]['vehicles'].append(vehicle)

        # return tour: list of c-v assignments
        tour = []
        for vehicle in self.state['warehouses'][id]['vehicles']:
            for customer in vehicle.customers:
                tour.append((customer.id, vehicle.id))
        return tour

    
    def vrp_l(self):
        pass
    
    def vrp_init(self, customers):
        # set up one agent for each warehouse
        # need to change this to only when there is no vehicle at the warehouse
        for warehouse in self.state['warehouses']:
            warehouse['vehicles'].append(Vehicle(len(warehouse['vehicles']), warehouse['location'], self.vehicle_cap))

        self.cluster_info = [{}, {}, {}, {}]

        for c in customers:
            if c.vehicle_id == -1 and c.assignment != 5: #!change
                c.arrival = self.env_time

        for i in range(4):
            clocs = [customer.location for customer in customers if customer.assignment == i + 1 and customer.cluster == None]
            c_idx = [customer.id for customer in customers if customer.assignment == i + 1 and customer.cluster == None] 
            #!change : gptprev asking to store customers in cidx??
            cluster_indices, centroids, radii, rho = self._vrp_cluster_gen(clocs, 10, self.state['warehouses'][i]['location'])
            for c in c_idx:
                customers[c].cluster = cluster_indices[c]
            self.cluster_info[i]['cluster_indices'] = cluster_indices
            self.cluster_info[i]['centroids'] = centroids
            self.cluster_info[i]['radii'] = radii
            self.cluster_info[i]['rho'] = rho


        # initialize the state for the vrp agent
        self.vrp_states = np.random.rand(4, 19) # placeholder for actual calculation
        self.vrp_actions = []
        for i in range(4):
            self.vrp_actions.append(self._compute_feasible_actions(i))


    def _vrp_cluster_gen(self, clocs, n, depot):
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
        diameters = [np.max([adj[i, j] for i in c for j in c]) for c in clusters]
        rho = np.max(diameters) / 2

        # generate cluster indices for each customer
        cluster_indices = np.zeros(ccount, dtype=int)
        for i, c in enumerate(clusters):
            for j in c:
                cluster_indices[j] = i

        # cluster centroids
        centroids = [np.mean(clocs[c], axis=0) for c in clusters]
        
        return cluster_indices, centroids, diameters/2, rho

    def compute_feasible_actions(self, vrp_id):
        # find the list of vehicles and customers
        vehicles = self.state['warehouses'][vrp_id]['vehicles']
        customers = [customer for customer in self.state['customers'] if customer.assignment == vrp_id + 1] # customer.vehicle_id == -1 !change

        # find the feasible actions
        # check distance between each vehicle and customer
        # check if the vehicle has enough capacity
        # check if the customer has not been assigned to a vehicle
        feasible_actions = []
        for vehicle in vehicles:
            for customer in customers:
                if (vehicle.current_cap + customer.demand <= vehicle.capacity and customer.vehicle_id == -1):
                    # and np.linalg.norm(np.array(vehicle.location) - np.array(customer.location)) / vehicle.speed
                    # <= customer.time_window[1] - self.clock):
                    dist = self.Euclidean_CV(customer.id, vrp_id, vehicle.id)
                    time = dist / vehicle.speed
                    if vehicle.customers == [] and time + self.env_time <= customer.time_window[1]:
                        feasible_actions.append((vehicle.id, customer.id))
                    elif vehicle.customers != [] and time + vehicle.time <= customer.time_window[1]:
                        feasible_actions.append((vehicle.id, customer.id))

        return feasible_actions


    def get_vrp_observation(self, action, id):
        vehicle_id, customer_id = action
        vehicle = self.state['warehouses'][id]['vehicles'][vehicle_id]
        customer = self.state['customers'][customer_id]

        # generate 17 length action vector

        d = self.Euclidean_CV(customer_id, id, vehicle_id)
        b_d_short = (d < customer.radius) # is d < neighborhood radius
        t = self.Euclidean_CV(customer_id, id, vehicle_id) / vehicle.speed # time taken to travel to c

        b_t_short = (t + vehicle.available <= customer.time_window[1]) # is t < time window end
        ngb = np.linalg.norm(vehicle.location - self.cluster_info[id]['centroids'][customer.cluster]) < self.cluster_info[id]['radii'][customer.cluster] # distance from vehicle to cluster centroid

        customer_list = [customer.id for customer in self.state['customers'] if customer.assignment == id + 1 and customer.arrival == self.env_time] # filter out previous time steps

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
                    if np.linalg.norm(np.array(cu.location) - 
                                      np.array(self.state['warehouses'][id]['location'])) > self.Euclidean_CV(
                                      cidx, id, vehicle_id):
                        drop_far = True

                if cu.vehicle_id == -1 and drop_cls:
                    if self.Euclidean_CV(cu.id, id, vehicle_id) < self.cluster_info[id]['rho'][cu.cluster]:
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
        remote = np.linalg.norm(np.array(customer.location) - np.array(self.cluster_info[id]['centroids'][customer.cluster]))/self.cluster_info[id]['radii'][customer.cluster]

        c_members_num = len([c for c in customer_list if self.state['customers'][c].cluster == customer.cluster])
        step = np.array([d/282.84, b_d_short, t/(282.84/vehicle.speed), b_t_short, ngb, non_d/282.84, c_left, drop_far, drop_cls, drop_long, served/vehicle.capacity, cls_dem, hops/c_members_num, cls_tim, urgt/(self.T/2), dfrac, remote], dtype=np.float32)
        return step

    def vrp_step(self, action, id):
        # here taking an action means assigning a customer to a vehicle
        # store vrp state before hand for rollout and the sort
        
        vehicle_id, customer_id = action
        vehicle = self.state['warehouses'][id]['vehicles'][vehicle_id]
        customer = self.state['customers'][customer_id]

        action_vector = self._get_vrp_observation(action, id)

        # update vehicle instance
        if vehicle.customers == []:
            vehicle.available = min(customer.time_window[0], self.env_time) + self.service_time
        else:
            vehicle.time += self.Euclidean_CV(customer_id, id, vehicle_id) / vehicle.speed + self.service_time
        vehicle.customers.append(customer)
        vehicle.current_cap += customer.demand
        customer.vehicle_id = vehicle_id
        vehicle.location = customer.location
        self.state['warehouses'][id]['vehicles'][vehicle_id] = vehicle
        # maybe dont do this here and do it explicitly when needed 

        # perform a state update
        self.vrp_states[id] = action_vector
        return action_vector
    
    def vrp_episode(self, i, epsilon):
        self.vrp_actions[i] = self.compute_feasible_actions(i)
        estimated_Qs = []

        # finding the top k actions
        for action in self.vrp_actions[i]:
            # save state and other details that change during action
            temp_state = self.vrp_states[i]
            vehicle_id, customer_id = action
            temp_vehicle = self.state['warehouses'][i]['vehicles'][vehicle_id]
            new_state = self.vrp_step(action, i)
            estimated_Qs.append(self.vrp_l(new_state))
            # restore the state
            self.vrp_states[i] = temp_state
            self.state['warehouses'][i]['vehicles'][vehicle_id] = temp_vehicle

        # pick the top k actions
        indices = np.argsort(estimated_Qs)[-self.kappa:]
        # compute distances while doing the rollout
        distances = [0 for _ in range(self.kappa)]
        paths = [[] for _ in range(self.kappa)]
        # store initial state before hand
        temp_init_state = self.vrp_states[i]
        temp_vehicles = self.state['warehouses'][i]['vehicles']

        # peform complete rollout for each of the top k actions
        for index in indices:
            paths[index].append(self.vrp_actions[i][index])
            self.vrp_step(self.vrp_actions[i][index], i)
            new_feasible_actions = self.compute_feasible_actions(i)

            while len(new_feasible_actions) != 0:
                estimated_Qs = []
                for action in new_feasible_actions:
                    temp_state = self.vrp_states[i]
                    vehicle_id, customer_id = action
                    temp_vehicle = self.state['warehouses'][i]['vehicles'][vehicle_id]
                    new_state = self.vrp_step(action, i)
                    estimated_Qs.append(self.vrp_l(new_state))
                    self.vrp_states[i] = temp_state
                    self.state['warehouses'][i]['vehicles'][vehicle_id] = temp_vehicle
                best_q_index = np.argmax(estimated_Qs)
                paths[index].append(new_feasible_actions[best_q_index])
                self.vrp_step(new_feasible_actions[best_q_index], i)
                new_feasible_actions = self.compute_feasible_actions(i)
        
        self.vrp_states[i] = temp_init_state
        self.state['warehouses'][i]['vehicles'] = temp_vehicles
        
        # need helper to compute the distance of the path
        for j in range(self.kappa):
            distances[j] = self.compute_distance(i, paths[j])
            
        # action with lowest distance is selected
        # implement epsilon greedy
        if random.random() < epsilon:
            index = random.randint(0, self.kappa - 1)
        else:
            index = np.argmin(distances)

        # use helper to get path in format for forward sat
        # returns a list of tours for each vehicle 
        optimized_tour = self.get_optimized_tour(i, paths[index]) 
        
        # should use optimised_sub_tour 
        self.vrp_step(paths[index][0], i)

        # check if this vehicle has any feasible actions left. If not, it leaves the depot and a new one is spawned
        vehicle_fesible = self.compute_feasible_actions(i)
        feasible = False
        for action in vehicle_fesible:
            vid, cid = action
            if vid == paths[index][0][0]: feasible = True
        if feasible == False:
            v = Vehicle(len(self.state['warehouses'][id]['vehicles']), self.state['warehouses'][id]['location'], self.vehicle_cap, self.env_time)
            self.state['warehouses'][i]['vehicles'].append(v)

        return optimized_tour


    # def compute_vrp_reward(self, optimized_tour, id):
    #     # organise into sub-tours
    #     tours = {}
    #     for action in optimized_tour:
    #         vehicle_id, customer_id = action
    #         if vehicle_id not in tours:
    #             tours[vehicle_id] = []
    #         tours[vehicle_id].append(customer_id)

    #     # compute the reward for each sub-tour
    #     rewards = []
    #     for vehicle_id, _ in tours.items():
    #         c_list = [self.state['customers'][c_id] for c_id in tours[vehicle_id]]
    #         d_p = []
    #         t_p = []
    #         d_max, t_max = 0, 0

    #         prev_loc = self.state['warehouses'][id]['location']
    #         for c in c_list:
    #             d_p.append(np.linalg.norm(np.array(prev_loc) - np.array(c.location)))
    #             t_p.append(d_p / self.state['warehouses'][id]['vehicles'][vehicle_id].speed)
    #             prev_loc = c.location

    #         # return to the warehouse
    #         d_p.append(np.linalg.norm(np.array(prev_loc) - np.array(self.state['warehouses'][vehicle_id]['location'])))
    #         t_p.append(d_p / self.state['warehouses'][id]['vehicles'][vehicle_id].speed)
    #         d_max = max(d_p)
    #         t_max = max(t_p)

    #         r = 0
    #         rho = self.cluster_info[id]['rho']
    #         r_term = (2 * rho) - (sum(d_p)*1/len(d_p))
    #         for i in range(len(c_list)):
    #             r += ((rho - d_p[i]) / d_max) + ((self.T - t_p[i]) / t_max) + (self.gamma**(len(c_list) - i) * r_term)
    #         rewards.append(r)

    #     # return the sum of the rewards
    #     return sum(rewards)


    def compute_vrp_reward(self, optimized_tour, id):
        # organise into sub-tours
        tours = {}
        for action in optimized_tour:
            vehicle_id, customer_id = action
            if vehicle_id not in tours:
                tours[vehicle_id] = []
            tours[vehicle_id].append(customer_id)

        # compute the reward for each sub-tour
        tuples = []
        for vehicle_id, _ in tours.items():
            # implement sub-tour
            vehicle_list = []

            # reinitialise vehicle
            self.state['warehouses'][id]['vehicles'][vehicle_id].location = self.state['warehouses'][id]['location']
            self.state['warehouses'][vehicle_id]['vehicles'][vehicle_id].current_cap = 0
            self.state['warehouses'][vehicle_id]['vehicles'][vehicle_id].customers = []
            self.state['warehouses'][vehicle_id]['vehicles'][vehicle_id].available = self.env_time

            c_list = [self.state['customers'][c_id] for c_id in tours[vehicle_id]]

            # calculate r_term for path
            d_p, t_p = [], []
            d_max, t_max = 0, 0
            prev_loc = self.state['warehouses'][id]['location']
            for c in c_list:
                d_p.append(np.linalg.norm(np.array(prev_loc) - np.array(c.location)))
                t_p.append(d_p / self.state['warehouses'][id]['vehicles'][vehicle_id].speed)
                prev_loc = c.location
            # return to the warehouse
            d_p.append(np.linalg.norm(np.array(prev_loc) - np.array(self.state['warehouses'][vehicle_id]['location'])))
            t_p.append(d_p / self.state['warehouses'][id]['vehicles'][vehicle_id].speed)
            d_max = max(d_p)
            t_max = max(t_p)
            rho = self.cluster_info[id]['rho']
            r_term = (2 * rho) - (sum(d_p)*1/len(d_p))

            # implement subtour to get (state, action, reward) tuple
            for i in range(len(c_list)):
                c = c_list[i]
                state = self.vrp_step((vehicle_id, c.id), id)
                reward = (rho - d_p[i]) / d_max + (self.T - t_p[i]) / t_max + (self.gamma**(len(c_list) - i) * r_term)
                vehicle_list.append((state, (vehicle_id, c.id), reward))
            tuples.append(vehicle_list)

        return tuples
     

    def env_step(self, epsilon_vrp):
        # execute the c2s agent
        # iterate through the list of unassigned customers and use c2s to decide the assignment

        c2s_tuples = []
        for order in self.orders.copy():
            state = self.get_c2s_observation()
            id = order.id #!change 
            if order.assignment == 0: #!change
                action = self.c2s_l()
                self.c2s_step(action)
                c2s_tuples.append((state, action, id))


        # execute the vrp agent
        customer_list = [customer for customer in self.state['customers'] if (customer.vehicle_id == -1 and customer.deferred != 5)]
        vrp_states = []
        vrp_actions = []
        for i in range(4):
            self.vrp_init(customer_list)
            # iterate while customers are left without vehicle assignment
            while len([customer for customer in customer_list if customer.assignment == i + 1 and customer.vehicle_id == -1]) > 0:
                optimized_tour = self.vrp_episode(i, epsilon_vrp)

        # compute reward using optimized_tour
        c2s_reward = self.compute_c2s_reward(optimized_tour)
        c2s_return = []
        for c in c2s_tuples:
            rew = c2s_reward[c[2]]
            c2s_return.append((c[0], c[1], rew))
        vrp_reward = self.compute_vrp_reward(optimized_tour)
        
        # increment environment time
        self.env_time += self.T

        # set the customers who've been deferred back to unasigned
        for order in self.orders:
            if order.assignment == 5: #!change
                order.assignment = 0 #!change
        
        # generate new customers
        num_customers = np.random.randint(200, 300)
        new_customers = [Customer(len(self.state['customers']) + i, arrival=self.env_time) for i in range(num_customers)] # !change
        self.orders += new_customers
        self.state['customers'] += new_customers 
        # self.gae_embeddings = np.random.rand(len(self.state['customers']), 2)

        clocs = np.array([c.location for c in self.state["customers"]])
        wlocs = np.array([w['location'] for w in self.state['warehouses']])
        customer_edges, rho = self._generate_edges(clocs, wlocs, n=5)

        customer_features = torch.tensor(clocs, dtype=torch.float)
        with torch.no_grad():
            self.gae_embeddings = self.gae_model.encode(customer_features, customer_edges).numpy()
        
        return c2s_return, vrp_reward

    def compute_distance(self, i, path):
        distance = np.linalg.norm(np.array(self.state['warehouses'][i]['location']) - np.array(self.state['customers'][path[0][1]].location))
        for j in range(1, len(path)):
            distance += self.Euclidean_CC(path[j-1][1], path[j][1])
        distance += np.linalg.norm(np.array(self.state['warehouses'][i]['location']) - np.array(self.state['customers'][path[-1][1]].location))
        return distance


    def Euclidean_CC(self, i, j):
        customer_i = self.state['customers'][i].location
        customer_j = self.state['customers'][j].location
        return np.linalg.norm(np.array(customer_i) - np.array(customer_j)) #!change

    def Euclidean_CV(self, c, w, v): # customer, warehouse, vehicle
        customer = self.state['customers'][c].location
        vehicle = self.state['warehouses'][w]['vehicles'][v].location
        return np.linalg.norm(np.array(customer) - np.array(vehicle)) #!change 