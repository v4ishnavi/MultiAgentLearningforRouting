import torch
import torch.nn.functional as F
import argparse
import logging
from torch_geometric.utils import to_dense_adj
from gae import GraphAutoEncoder
from datagen import generate_data
from torch_geometric.loader import DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(model, data_loader, num_epochs=100, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data in data_loader:
            optimizer.zero_grad()
            adj_hat = model(data.x, data.edge_index)
            
            adj_true = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]

            # unnecessary step : but just incase 
            adj_hat = torch.sigmoid(adj_hat)
            adj_true = torch.sigmoid(adj_true)
            
            loss = F.binary_cross_entropy(adj_hat, adj_true)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        logger.info(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train Graph AutoEncoder Model')
    parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension size for the GCN layers')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train the model')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for the optimizer')
    parser.add_argument('--train_size', type=int, default=10, help='Number of graphs to generate for training')
    parser.add_argument('--eval_size', type=int, default=2, help='Number of graphs to generate for evaluation')
    args = parser.parse_args()
    
    input_dim = 2
    hidden_dim = args.hidden_dim
    output_dim = 2
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    train_size = args.train_size
    batch_size = 1
    # diff graphs have diff number of nodes, better to set as 1 
    eval_size = args.eval_size
    
    model = GraphAutoEncoder(input_dim, hidden_dim, output_dim)
    
    graph_list_train = [generate_data() for _ in range(train_size)]
    data_loader = DataLoader(graph_list_train, batch_size=batch_size, shuffle=True)
    
    train_model(model, data_loader, num_epochs=num_epochs, lr=learning_rate)
    
    model.eval()
    graph_list_eval = [generate_data() for _ in range(eval_size)]
    for i, data in enumerate(graph_list_eval):
        embeddings = model.encode(data.x, data.edge_index)
        logger.info(f"Embeddings for graph {i+1}: {embeddings}")