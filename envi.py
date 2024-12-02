import numpy as np 

class Customer():
    def __init__(self, id, T=100, arrival):
        self.id = id
        self.location = (np.random.uniform(-100, 100), np.random.uniform(-100, 100))
        self.demand = np.random.randint(1, 11)
        self.time_window = (np.random.randint(T // 5, 4 *(T //5)), 0)
        self.time_window = (self.time_window[0], self.time_window[0] + np.random.randint(T // 10, 2 * T))
        self.assignment = 0 # 0:unassigned, 1-4: assigned to warehouse 1-4
        self.deferred = 0
        self.vehicle_id = -1 # -1: not assigned, >=0 assigned to vehicle
        self.arrival = arrival
        
        self.cluster = None
        self.served = False

class Vehicle():
    def __init__(self, id, location, capacity):
        self.id = id
        self.location = location
        self.capacity = capacity
        self.current_cap = 0
        self.customers = []
        self.speed = 2

        self.time = 0 # time taken for the vehicle to satisfy customers in current sub-tour

class CV_pair():
    def __init__(self, customer_list, vehicle_list, c, v):
        customer = customer_list[c]
        vehicle = vehicle_list[v]

        self.customer_id = customer.id
        self.vehicle_id = vehicle.id

        self.d = np.linalg.norm(np.array(customer.location) - np.array(vehicle.location)) # distance from loc to c
        self.b_d_short = (self.d < customer.radius) # is d < neighborhood radius
        self.t = np.linalg.norm(np.array(customer.location) - np.array(vehicle.location))/vehicle.speed # time taken to travel to c

        self.b_t_short = (self.t <= customer.time_window[1] ) # - self.clock (current time) # is t < time window start
        self.ngb = np.linalg.norm(vehicle.location - centroids[customer.cluster]) < radii[customer.cluster] # distance from vehicle to cluster centroid

        if self.ngb:
            self.non_d = np.inf # distance from c to nearest non-member
            for cu in customer_list:
                if cu.cluster != customer.cluster:
                    self.non_d = min(self.non_d, np.linalg.norm(np.array(cu.location) - np.array(vehicle.location)))
            self.c_left = 0
        else:
            self.non_d = np.inf
            self.c_left = 0
            for cu in customer_list:
                if cu.cluster == customer.cluster and cu.vehicle_id == -1:
                    self.c_left += 1
                    break
            
        if self.c_left == 1:

        # self.drop_far = dropped customers are vehicle.customers?
        # self.drop_cls
        # self.drop_long = how to calculate the distance between one point and many points?
        self.served = len(vehicle.customers) # what if a customer's time window was missed? are they still counted as served?
        # self.cls_dem = (vehicle.capacity - vehicle.current_cap < remaining sum of demands of cluster)
        # self.hops = time window-wise?
        # self.cls_tim = define feasible
        # self.urgt = again, what is the current time for the vehicle?
        # self.dfrac = time being used to seve c (what is this) / (vehicle.curent_cap/vehicle.capacity)
        # self.remote = what is this?


# not using this class
class VRP_Agent():
    def __init__(self, id):
        self.depot = id # warehouse number
        self.state = np.random.rand(19,) # need some function to compute the state
        self.feasible_actions = None

        self.customers = []
        self.vehicles = []
        # cluster the customers now?
        # self.radius = r

    def get_feasible_actions(self):
        '''
        each time do we entirely redefine the feasible actions?
        because if v is assigned a new customer, evey VC pair with that V changes
        and every VC pair with that C is deleted

        we need to know the time a vehicle is at. Meaning, given their current sub-tour, what is the current time given their current customer list
        when do we add return times here? because the notion of vehicles returning to depot must exist, but is not accounted for here
        '''
        pass
        

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
        self.kappa = 2
        self.env_time = 0
        self.service_time = 1
    
    def initialize_environment(self):
        num_customers = np.random.randint(200, 301)
        env_info = {
            "warehouses": [
                {"location": (50, 50), "inventory": self.P_0max, "vehicles": []},
                {"location": (50, -50), "inventory": self.P_0max, "vehicles": []},
                {"location": (-50, -50), "inventory": self.P_0max, "vehicles": []},
                {"location": (-50, 50), "inventory": self.P_0max, "vehicles": []},
            ],
            "customers": [Customer(i, arrival=0)for i in range(num_customers)],
        }
        for customer in env_info['customers']:
            customer['time_window'] = (customer['time_window'][0], customer['time_window'][0] + np.random.randint(self.T // 10, 2 * self.T))

        # init gae_embeddings
        self.gae_embeddings = np.random.rand(len(env_info['customers']), 2)
        self.env_info = env_info   
        self.orders = [env_info['customers'][i] for i in range(len(env_info['customers']))] 
        
        return env_info
    
    def time_step(self):
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
        
    def get_c2s_observation(self):
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
    
    def vrp_init(self):
        # set up one agent for each warehouse
        for warehouse in self.state['warehouses']:
            warehouse['vehicles'].append(Vehicle(len(warehouse['vehicles']), warehouse['location'], self.vehicle_cap))

        self.cluster_info = [{}, {}, {}, {}]

        for i in range(4):
            clocs = [customer.location for customer in self.state['customers'] if customer.assignment == i + 1 and customer.cluster == None]
            c_idx = [customer.id for customer in self.state['customers'] if customer.assignment == i + 1 and customer.cluster == None]
            cluster_indices, centroids, radii, rho = self._vrp_cluster_gen(clocs, 10, self.state['warehouses'][i]['location'])
            for c in c_idx:
                self.state['customers'][c].cluster = cluster_indices[c]
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
            while True:
                old_len = len(new_cluster)
                for c in new_cluster:
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


    def _compute_feasible_actions(self, vrp_id):
        # find the list of vehicles and customers
        vehicles = self.state['warehouses'][vrp_id]['vehicles']
        customers = [customer for customer in self.state['customers'] if customer['assignment'] == vrp_id + 1] # customer.vehicle_id == -1

        # find the feasible actions
        # check distance between each vehicle and customer
        # check if the vehicle has enough capacity
        # check if the customer has not been assigned to a vehicle
        feasible_actions = []
        for vehicle in vehicles:
            for customer in customers:
                if (vehicle.current_cap + customer.demand <= vehicle.capacity and customer.vehicle_id == -1
                    and np.linalg.norm(np.array(vehicle.location) - np.array(customer.location)) / vehicle.speed
                    <= customer.time_window[1] - self.clock):
                    feasible_actions.append((vehicle.id, customer.id))
        return feasible_actions
    
    def get_vrp_observation(self, action, id):
        vehicle_id, customer_id = action
        vehicle = self.state['warehouses'][id]['vehicles'][vehicle_id]
        customer = self.state['customers'][customer_id]

        # generate 17 length action vector
        # d = np.linalg.norm(np.array(customer.location) - np.array(vehicle.location)) # distance from loc to c
        d = self.Euclidean_CV(customer_id, id, vehicle_id)
        b_d_short = (d < customer.radius) # is d < neighborhood radius
        t = np.linalg.norm(np.array(customer.location) - np.array(vehicle.location))/vehicle.speed # time taken to travel to c

        b_t_short = (t <= customer.time_window[1] - vehicle.time) # is t < time window end
        ngb = np.linalg.norm(vehicle.location - self.cluster_info[id]['centroids'][customer.cluster]) < self.cluster_info[id]['radii'][customer.cluster] # distance from vehicle to cluster centroid

        # THIS CUSTOMER_LIST IF WRONG
        customer_list = [customer.id for customer in self.state['customers'] if customer.assignment == id + 1 and customer.arrival = self.env_time] # filter out previous time steps

        c_left = False
        non_d = np.inf # distance from c to nearest non-member
        cust_non_d = None
        for cidx in customer_list:
            cu = self.state['customers'][cidx]
            if cu.cluster != customer.cluster:
                # non_d = min(non_d, np.linalg.norm(np.array(cu.location) - np.array(vehicle.location)))
                if ngb:
                    non_d = min(non_d, self.Euclidean_CV(cu.id, id, vehicle_id))
                else:
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
        cls_dem = (vehicle.capacity - vehicle.current_cap < remaining_demand)


        # hops How many cluster members of c can be served before c
        # cls_tim Is every cluster member feasible following c
        hops = 0
        cls_tim = True
        for cidx in customer_list:
            cu = self.state['customers'][cidx]
            if cu.cluster == customer.cluster and cu.vehicle_id == -1:
                distance = self.Euclidean_CV(cu.id, id, vehicle_id) + self.Euclidean_CC(customer_id, cu.id)
                time = distance / vehicle.speed + self.service_time
                if customer.time_window[1] - vehicle.time >= time:
                    hops += 1

                distance = d + self.Euclidean_CC(customer_id, cu.id)
                time = distance / vehicle.speed + self.service_time
                if cu.time_window[1] - vehicle.time >= time:
                    cls_tim = False

        # urgt How close to time window closure of c is v arriving
        urgt = customer.time_window[1] - (vehicle.time + t)

        dfrac = t + self.service_time / (vehicle.curent_cap + customer.demand/vehicle.capacity)
        remote = np.linalg.norm(np.array(customer.location) - np.array(self.cluster_info[id]['centroids'][customer.cluster]))

        step = np.array([d, b_d_short, t, b_t_short, ngb, non_d, c_left, drop_far, drop_cls, drop_long, served, cls_dem, hops, cls_tim, urgt, dfrac, remote], dtype=np.float32)
        return step

    def vrp_step(self, action, id):
        # here taking an action means assigning a customer to a vehicle
        # store vrp state before hand for rollout and the sort
        
        vehicle_id, customer_id = action
        vehicle = self.state['warehouses'][id]['vehicles'][vehicle_id]
        customer = self.state['customers'][customer_id]



        # update vehicle instance
        vehicle.customers.append(customer)
        vehicle.current_cap += customer.demand
        customer.vehicle_id = vehicle_id
        vehicle.location = customer.location
        # maybe dont do this here and do it explicitly when needed 

        # self.vrp_actions[id] = self._compute_feasible_actions(id)
        # perform a state update
        return self._get_vrp_observation()
    
    def vrp_episode(self):
        for i in range(4): # shouldn't be here
            self.vrp_actions[i] = self.compute_feasible_actions(i)
            estimated_Qs = []

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
            # store initial state before hand
            # compute distances while doing the rollout
            distances = [0 for _ in range(self.kappa)]
            paths = [[] for _ in range(self.kappa)]
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
                distances[j] = self.compute_distance(paths[j])
                
            
            # action with lowest distance is selected
            index = np.argmin(distances)
            
            # use helper to get path in format for forward sat
            optimized_tour = self.get_optimized_tour(i, paths[index]) 
            
            self.vrp_step(paths[index][0], i)
                
                
            
    
    def env_step(self):
        # execute the c2s agent
        # iterate through the list of unassigned customers and use c2s to decide the assignment
        for order in self.orders:
            if order['assignment'] == 0:
                action = self.c2s_h()
                self.c2s_step(action)
        
        pass
    
    def Euclidean_CC(self, i, j):
        customer_i = self.state['customers'][i].location
        customer_j = self.state['customers'][j].location
        return np.linalg.norm(np.array(customer_i.location) - np.array(customer_j.location))

    def Euclidean_CV(self, c, w, v): # customer, warehouse, vehicle
        customer = self.state['customers'][c].location
        vehicle = self.state['warehouses'][w]['vehicles'][v].location
        return np.linalg.norm(np.array(customer.location) - np.array(vehicle.location))