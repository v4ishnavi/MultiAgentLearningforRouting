import numpy as np
import torch
from torch_geometric.data import Data
import torch_geometric.utils as utils

def rho_adj_matrix(clocs, n, wlocs):
    ccount = len(clocs)
    adj = np.zeros((ccount, ccount))
    # print(f"Number of customers: {ccount}")
    # Compute distances between all customers
    for i in range(ccount):
        for j in range(ccount):
            adj[i, j] = np.linalg.norm(clocs[i] - clocs[j])
    # print("Computed distances between all customers")
    depot_dists = np.array([min([np.linalg.norm(clocs[i] - w) for w in wlocs]) for i in range(ccount)])
    # Form clusters based on nearest neighbors to warehouses

    assigned = np.zeros(ccount, dtype=bool)
    clusters = []
    # print("Forming clusters based on nearest neighbors to warehouses")
    while not np.all(assigned): # Assign all nodes to clusters
        new_cluster = []
        unassigned = np.where(~assigned)[0]
        nearest = unassigned[np.argmin(depot_dists[unassigned])] # Find nearest unassigned node to warehouse
        new_cluster.append(nearest) #
        assigned[nearest] = True # Assign nearest node to cluster
        # print("Assigned all nodes to clusters")

        temp = new_cluster.copy()
        # print(f"Temp: {temp}")
        for c in temp: # For each node in cluster
            old_len = len(temp)
            unassigned = np.where(~assigned)[0]
            nearest_neighbors = unassigned[np.argsort([adj[c, i] for i in unassigned])[:n]]
            # nearest neighbors to node c which are unassigned and not in cluster 
            # print(f"Nearest neighbors: {nearest_neighbors}")
            # print("#"*50)
            # iter = 1
            for nn in nearest_neighbors:
                if not assigned[nn]: # Assign nearest neighbors to cluster
                    new_cluster.append(nn) # Add nearest neighbor to cluster
                    assigned[nn] = True # Assign nearest neighbor to cluster
            # iter +=1
            if len(new_cluster) == old_len: # If no new nodes were added to cluster
                break
            # if iter == 2: 
            #     break
        
        clusters.append(new_cluster)

    # Compute cluster diameters and determine rho
    # for c in clusters:
    #     print(c)
    # print(f"Number of clusters: {len(clusters)}")
    diameters = [max([adj[i, j] for i in c for j in c]) for c in clusters]
    # print("Computed cluster diameters")
    # print(f"Cluster diameters: {diameters}")
    rho = np.mean(diameters) / 2  # Maximum haf-diameter
    # print("Computed cluster diameters and determined rho")
    # print(f"Maximum half-diameter: {rho:.2f}")
    print("-------------------------------------------------------------------------------")
    # Form adjacency matrix with nodes within `rho` of a warehouse
    count_positives = 0 # count number of positives in adj matrix
    final_adj = np.zeros((ccount, ccount))
    for i in range(ccount):
        for j in range(ccount):
            # connect nodes which are within rho distance of a warehouse
            if np.any([np.linalg.norm(clocs[i] - w) < rho for w in wlocs]) and np.any([np.linalg.norm(clocs[j] - w) < rho for w in wlocs]):
                final_adj[i, j] = 1
                count_positives += 1

    # for number of positives, we make equal number negative samples randomly
    count_negatives = 0
    # print(f"count_positives: {count_positives}")
    # print(f"count total: {ccount*ccount}")
    
    # print("Formed adjacency matrix with nodes within `rho` of a warehouse")
    while count_negatives < count_positives:
        i, j = np.random.randint(0, ccount), np.random.randint(0, ccount)
        if adj[i, j] > rho and final_adj[i, j] == 0:
            final_adj[i, j] = -1
            count_negatives += 1
    
    # Convert adjacency matrix to sparse edge indices
    edges = utils.dense_to_sparse(torch.tensor(final_adj, dtype=torch.float))[0]
    # print("Converted adjacency matrix to sparse edge indices")
    print("-------------------------------------------------------------------------------")
    return edges, rho

def generate_data():
    """
    Generates a graph with adjacency matrix based on `rho` clustering.

    Returns:
        torch_geometric.data.Data: Graph data containing nodes and edges.
    """
    x_min, x_max = -100, 100
    y_min, y_max = -100, 100
    wlocs = [(-50, 50), (50, 50), (-50, -50), (50, -50)]  # Predefined warehouse locations

    ccount = np.random.randint(200,300)  # Randomly generate number of customers
    clocs = np.random.uniform(low=x_min, high=x_max, size=(ccount, 2))  # Generate customer locations

    # Generate adjacency matrix and rho using clustering algorithm
    edges, rho = rho_adj_matrix(clocs, n=5, wlocs=wlocs)
    nodes = torch.tensor(clocs, dtype=torch.float)  # Node features are customer locations
    print(f"Generated graph with rho: {rho:.2f}")
    return Data(x=nodes, edge_index=edges)
