import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
import numpy as np
from torch_geometric.nn.conv import MessagePassing

def scatter_mean(src, index, dim=-1, out=None, dim_size=None):
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = src.new_zeros(size)

    out.scatter_add_(dim, index, src)
    count = torch.zeros_like(out)
    count.scatter_add_(dim, index, torch.ones_like(src))
    count[count == 0] = 1
    return out / count

def scatter_sum(src, index, dim=-1, out=None, dim_size=None):
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = src.new_zeros(size)

    return out.scatter_add_(dim, index, src)

def scatter_softmax(src, index, dim=-1):
    if index.numel() == 0:
        return src

    max_value_per_index = torch.zeros_like(src)
    max_value_per_index.scatter_reduce_(dim, index, src, reduce='amax')
    max_value_per_index = max_value_per_index.index_select(dim, index)

    recentered_scores = src - max_value_per_index
    exp_scores = torch.exp(recentered_scores)

    sum_per_index = torch.zeros_like(src)
    sum_per_index.scatter_add_(dim, index, exp_scores)
    sum_per_index = sum_per_index.index_select(dim, index)

class EnhancedMPNNConv(MessagePassing):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__(aggr='add')
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Node feature transformation
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),  # SiLU activation for better gradient flow
            nn.Dropout(dropout)
        )
        
        # Edge feature transformation
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-head edge attention
        self.edge_attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 3, self.head_dim),
                nn.LayerNorm(self.head_dim),
                nn.SiLU(),
                nn.Linear(self.head_dim, 1)
            ) for _ in range(num_heads)
        ])
        
        # Multi-head node attention
        self.node_attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, self.head_dim),
                nn.LayerNorm(self.head_dim),
                nn.SiLU(),
                nn.Linear(self.head_dim, 1)
            ) for _ in range(num_heads)
        ])
        
        # Message MLPs (one per head)
        self.message_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 3, self.head_dim),
                nn.LayerNorm(self.head_dim),
                nn.SiLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_heads)
        ])
        
        # Update MLP
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        # Virtual node projection
        self.virtual_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        # Output transformation
        self.output_transform = nn.Sequential(
            nn.Linear(hidden_dim * num_heads, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, edge_index, edge_attr, batch=None):
        # Initial node and edge features transformation
        edge_features = self.edge_mlp(edge_attr)
        
        # Multi-head processing
        head_outputs = []
        attention_weights = []
        
        for head in range(self.num_heads):
            # Process messages for this head
            head_out = self.propagate(
                edge_index, 
                x=x,
                edge_features=edge_features,
                head_idx=head
            )
            head_outputs.append(head_out)
            
            # Calculate attention weights for visualization
            src, dst = edge_index
            edge_attention = self.edge_attention[head](
                torch.cat([x[src], x[dst], edge_features], dim=-1)
            )
            attention_weights.append(edge_attention)
        
        # Combine head outputs
        combined_output = torch.cat(head_outputs, dim=-1)
        output = self.output_transform(combined_output)
        
        # Store attention weights for visualization
        self.last_attention = torch.cat(attention_weights, dim=-1)
        
        return output
    
    def message(self, x_i, x_j, edge_features, head_idx: int):
        # Combine node and edge features
        msg_input = torch.cat([x_i, x_j, edge_features], dim=-1)
        
        # Calculate message using head-specific MLP
        message = self.message_mlps[head_idx](msg_input)
        
        # Apply edge attention
        edge_attention = torch.sigmoid(
            self.edge_attention[head_idx](msg_input)
        )
        
        return message * edge_attention
    
    def update(self, aggr_out, x):
        # Combine aggregated messages with node features
        combined = torch.cat([aggr_out, x], dim=-1)
        return self.update_mlp(combined)

class UncertaintyMPNN(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, 
                 num_layers: int, num_tasks: int, dropout_rate: float = 0.1,
                 num_heads: int = 4):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_tasks = num_tasks
        
        # Initial node embedding
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        
        # Stack of MPNN layers
        self.convs = nn.ModuleList([
            EnhancedMPNNConv(
                node_dim=hidden_dim,
                edge_dim=edge_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout_rate
            ) for _ in range(num_layers)
        ])
        
        # Global pooling attention
        self.pool_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Output layers for mean prediction
        self.output_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_tasks)
        )
        
        # Output layers for uncertainty prediction
        self.output_logvar = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_tasks)
        )
        
        # Layer for virtual node
        self.virtual_node = nn.Parameter(torch.zeros(1, hidden_dim))
        
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Initial node embeddings
        x = self.node_embedding(x)
        
        # Store attention weights for interpretation
        self.attention_weights = []
        
        # Message passing layers
        for conv in self.convs:
            # Add virtual node to all nodes
            batch_size = batch.max().item() + 1
            virtual_node_expanded = self.virtual_node.expand(batch_size, -1)
            virtual_node_mapped = virtual_node_expanded[batch]
            
            # Update node features
            x_updated = conv(x + virtual_node_mapped, edge_index, edge_attr, batch)
            
            # Store attention weights
            self.attention_weights.append(conv.last_attention)
            
            # Update virtual node
            virtual_node_new = scatter_mean(x_updated, batch, dim=0)
            self.virtual_node.data = virtual_node_new.mean(dim=0, keepdim=True)
            
            # Residual connection
            x = x + x_updated
        
        # Global pooling with attention
        pool_weights = self.pool_attention(x)
        pool_weights = scatter_softmax(pool_weights, batch, dim=0)
        
        # Weighted global pooling
        global_repr = scatter_sum(x * pool_weights, batch, dim=0)
        
        # Predict mean and log variance
        mean = self.output_mean(global_repr)
        logvar = self.output_logvar(global_repr)
        
        return mean, logvar
    
    def predict_with_uncertainty(self, data, num_samples: int = 10):
        """Make predictions with uncertainty estimation using MC Dropout."""
        self.train()  # Enable dropout
        
        predictions = []
        for _ in range(num_samples):
            mean, logvar = self(data)
            predictions.append(mean.unsqueeze(0))
        
        # Calculate statistics
        predictions = torch.cat(predictions, dim=0)
        mean_pred = predictions.mean(dim=0)
        epistemic_uncertainty = predictions.var(dim=0)  # Model uncertainty
        aleatoric_uncertainty = torch.exp(logvar)  # Data uncertainty
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        return mean_pred, total_uncertainty
    
    def get_attention_weights(self, data):
        """Get attention weights for interpretation."""
        self.eval()
        with torch.no_grad():
            _ = self(data)
            return {
                'edge_attention': torch.stack(self.attention_weights),
                'pool_attention': self.pool_attention(data.x)
            }

class MolecularGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels, num_tasks):
        super(MolecularGNN, self).__init__()
        
        # GNN layers
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        
        # Output layers
        self.lin1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, num_tasks)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, edge_index, batch):
        # Initial convolutions
        x1 = F.relu(self.bn1(self.conv1(x, edge_index)))
        x1 = self.dropout(x1)
        
        x2 = F.relu(self.bn2(self.conv2(x1, edge_index)))
        x2 = self.dropout(x2)
        
        x3 = F.relu(self.bn3(self.conv3(x2, edge_index)))
        x3 = self.dropout(x3)
        
        # Global pooling
        x_mean = global_mean_pool(x3, batch)
        x_sum = global_add_pool(x3, batch)
        x = torch.cat([x_mean, x_sum], dim=1)
        
        # MLP for final prediction
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)
        
        return x

def smiles_to_graph(smiles):
    """Convert SMILES to PyTorch Geometric graph data."""
    mol = Chem.MolFromSmiles(smiles)
    
    # Get node features
    node_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetNumRadicalElectrons(),
            int(atom.GetIsAromatic()),
            atom.GetHybridization(),
            atom.GetTotalNumHs()
        ]
        node_features.append(features)
    
    # Get edge indices and features
    edge_indices = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        # Add edges in both directions
        edge_indices += [[i, j], [j, i]]
        
        # Bond features
        features = [
            bond.GetBondTypeAsDouble(),
            int(bond.GetIsConjugated()),
            int(bond.IsInRing())
        ]
        edge_features += [features, features]
    
    # Convert to PyTorch tensors
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

class PolymerGNNModel:
    def __init__(self, num_tasks=5, hidden_channels=128, device='cuda'):
        self.num_tasks = num_tasks
        self.hidden_channels = hidden_channels
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Initialize model
        self.model = MolecularGNN(
            num_node_features=7,  # Number of atom features
            num_edge_features=3,  # Number of bond features
            hidden_channels=hidden_channels,
            num_tasks=num_tasks
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
    
    def train_step(self, loader):
        self.model.train()
        total_loss = 0
        
        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            out = self.model(batch.x, batch.edge_index, batch.batch)
            loss = self.criterion(out, batch.y)
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        
        return total_loss
    
    def predict(self, smiles_list):
        self.model.eval()
        graphs = [smiles_to_graph(s) for s in smiles_list]
        loader = DataLoader(graphs, batch_size=32)
        
        predictions = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.edge_index, batch.batch)
                predictions.append(out.cpu().numpy())
        
        return np.vstack(predictions) 