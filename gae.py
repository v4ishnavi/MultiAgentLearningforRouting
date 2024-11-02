import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
class GraphAutoEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphAutoEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        z = self.conv2(x, edge_index)
        return z

    def decode(self, z):
        norm_diff = torch.cdist(z, z, p=2)
        max_diff = norm_diff.max() if norm_diff.max() != 0 else 1 
        similarity = 1 - (norm_diff / max_diff) 
        adj_hat = similarity  # can add sigmoid later? 
        return adj_hat

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        adj_hat = self.decode(z)
        return adj_hat