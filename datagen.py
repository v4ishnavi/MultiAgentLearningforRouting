import torch
import numpy as np
from torch_geometric.data import Data
import torch_geometric.utils as utils

def generate_data():
    x_min, x_max = -100, 100
    y_min, y_max = -100, 100
    wlocs = [(-50, 50), (50, 50), (-50, -50), (50, -50)]

    ccount = np.random.randint(200, 300)
    clocs = np.random.uniform(low=x_min, high=x_max, size=(ccount, 2))

    clusters = []
    for c in clocs:
        d = [np.linalg.norm(c - np.array(wh)) for wh in wlocs]
        clusters.append(np.argmin(d))
    
    adj = np.zeros((ccount, ccount))
    for i, _ in enumerate(clusters):
        for j in range(ccount):
            if i != j and clusters[i] == clusters[j]:
                adj[i, j] = 1
    
    nodes = torch.tensor(clocs, dtype=torch.float)
    
    edges = utils.dense_to_sparse(torch.tensor(adj, dtype=torch.float))[0]
    return Data(x=nodes, edge_index=edges)