import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
from typing import List, Tuple, Optional, Dict

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        
        # Linear projections and split into heads
        q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Attention weights and weighted sum
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        
        return self.out(out), attn_weights

class EnhancedMPNNConv(MessagePassing):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, num_heads: int = 4):
        super().__init__(aggr='add')
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        
        # Node feature transformation
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Edge feature transformation
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Edge attention
        self.edge_attention = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Multi-head attention for node updates
        self.node_attention = MultiHeadAttention(hidden_dim, num_heads)
        
        # Message MLP
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Update MLP
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Virtual node projection
        self.virtual_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, virtual_node: torch.Tensor,
                batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # Virtual node broadcasting
        x = x + virtual_node[batch]
        
        # Transform edge features
        edge_features = self.edge_mlp(edge_attr)
        
        # Message passing
        out, attention_weights = self.propagate(edge_index, x=x, 
                                              edge_features=edge_features,
                                              return_attention=True)
        
        # Update virtual node
        virtual_node_temp = global_add_pool(out, batch)
        virtual_node = virtual_node + self.virtual_mlp(virtual_node_temp)
        
        return out, virtual_node, attention_weights
        
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor,
                edge_features: torch.Tensor) -> torch.Tensor:
        # Compute edge attention weights
        edge_attention_input = torch.cat([x_i, x_j, edge_features], dim=-1)
        edge_attention_weights = torch.sigmoid(self.edge_attention(edge_attention_input))
        
        # Construct messages
        message_input = torch.cat([x_i, x_j, edge_features], dim=-1)
        message = self.message_mlp(message_input)
        
        # Apply edge attention
        message = message * edge_attention_weights
        
        return message, edge_attention_weights
    
    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Apply multi-head attention
        attended_features, node_attention_weights = self.node_attention(
            torch.cat([x, aggr_out], dim=-1).unsqueeze(0)
        )
        attended_features = attended_features.squeeze(0)
        
        # Update node embeddings
        update_input = torch.cat([x, attended_features], dim=-1)
        updated_features = self.update_mlp(update_input)
        
        return updated_features, node_attention_weights

class EnhancedUncertaintyMPNN(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int,
                 num_layers: int, num_tasks: int, num_heads: int = 4,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.num_tasks = num_tasks
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        
        # Initial node embedding
        self.node_embedding = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Virtual node embedding
        self.virtual_node_embedding = nn.Parameter(torch.zeros(1, hidden_dim))
        
        # Message passing layers
        self.conv_layers = nn.ModuleList([
            EnhancedMPNNConv(hidden_dim, edge_dim, hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Global attention pooling
        self.global_attention = MultiHeadAttention(hidden_dim, num_heads)
        
        # Output layers for mean prediction
        self.output_mean = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_tasks)
        )
        
        # Output layers for uncertainty (log variance)
        self.output_logvar = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_tasks)
        )
        
    def forward(self, data: Batch) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Initial embeddings
        x = self.node_embedding(x)
        virtual_node = self.virtual_node_embedding.expand(data.num_graphs, -1)
        
        # Store attention weights
        attention_weights = {
            'edge': [],
            'node': [],
            'global': None
        }
        
        # Message passing layers
        for conv in self.conv_layers:
            x_residual = x
            x, virtual_node, layer_attention = conv(x, edge_index, edge_attr,
                                                  virtual_node, batch)
            x = x + x_residual  # Residual connection
            
            attention_weights['edge'].append(layer_attention['edge'])
            attention_weights['node'].append(layer_attention['node'])
        
        # Global attention pooling
        x_graph = global_mean_pool(x, batch)
        x_graph = x_graph + virtual_node
        
        x_attended, global_attn = self.global_attention(x_graph.unsqueeze(0))
        x_attended = x_attended.squeeze(0)
        attention_weights['global'] = global_attn
        
        # Concatenate different pooling results
        pooled = torch.cat([x_attended, x_graph], dim=-1)
        
        # Predict mean and log variance
        mean = self.output_mean(pooled)
        logvar = self.output_logvar(pooled)
        
        return mean, logvar, attention_weights
    
    def predict_with_uncertainty(self, data: Batch,
                               num_samples: int = 30) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """Make predictions with uncertainty estimation using MC Dropout."""
        self.train()  # Enable dropout
        
        predictions = []
        attention_weights_samples = []
        
        for _ in range(num_samples):
            mean, logvar, attention = self(data)
            predictions.append((mean.detach().cpu().numpy(),
                             logvar.detach().cpu().numpy()))
            attention_weights_samples.append({
                k: v.detach().cpu().numpy() if v is not None else None
                for k, v in attention.items()
            })
        
        # Calculate mean and uncertainty
        means = np.stack([p[0] for p in predictions])
        logvars = np.stack([p[1] for p in predictions])
        
        pred_mean = np.mean(means, axis=0)
        pred_var = np.exp(np.mean(logvars, axis=0)) + np.var(means, axis=0)
        pred_std = np.sqrt(pred_var)
        
        # Average attention weights
        avg_attention = {
            'edge': np.mean([s['edge'] for s in attention_weights_samples], axis=0),
            'node': np.mean([s['node'] for s in attention_weights_samples], axis=0),
            'global': np.mean([s['global'] for s in attention_weights_samples], axis=0)
        }
        
        return pred_mean, pred_std, avg_attention

class EnhancedMPNNModel:
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 128,
                 num_layers: int = 4, num_tasks: int = 5, num_heads: int = 4,
                 device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = EnhancedUncertaintyMPNN(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_tasks=num_tasks,
            num_heads=num_heads
        ).to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=0.01
        )
        
    def train_step(self, batch: Batch) -> Tuple[float, Dict[str, torch.Tensor]]:
        """Train the model for one step."""
        self.model.train()
        batch = batch.to(self.device)
        
        # Forward pass
        mean, logvar, attention_weights = self.model(batch)
        
        # Negative log likelihood loss with uncertainty
        loss = self.gaussian_nll_loss(mean, logvar, batch.y)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), attention_weights
    
    def predict(self, data_list: List[Data], batch_size: int = 32,
                return_uncertainty: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict[str, np.ndarray]]]:
        """Make predictions with optional uncertainty estimation."""
        self.model.eval()
        predictions = []
        uncertainties = []
        attention_weights = []
        
        # Process in batches
        for i in range(0, len(data_list), batch_size):
            batch_data = Batch.from_data_list(
                data_list[i:i + batch_size]
            ).to(self.device)
            
            if return_uncertainty:
                pred_mean, pred_std, attn = self.model.predict_with_uncertainty(batch_data)
                predictions.append(pred_mean)
                uncertainties.append(pred_std)
                attention_weights.append(attn)
            else:
                with torch.no_grad():
                    mean, _, attn = self.model(batch_data)
                    predictions.append(mean.cpu().numpy())
                    attention_weights.append({
                        k: v.cpu().numpy() if v is not None else None
                        for k, v in attn.items()
                    })
        
        # Combine results
        predictions = np.vstack(predictions)
        
        if return_uncertainty:
            uncertainties = np.vstack(uncertainties)
            # Combine attention weights
            combined_attention = {
                'edge': np.concatenate([a['edge'] for a in attention_weights], axis=0),
                'node': np.concatenate([a['node'] for a in attention_weights], axis=0),
                'global': np.concatenate([a['global'] for a in attention_weights], axis=0)
            }
            return predictions, uncertainties, combined_attention
        
        return predictions, None, None
    
    @staticmethod
    def gaussian_nll_loss(mean: torch.Tensor, logvar: torch.Tensor,
                         target: torch.Tensor) -> torch.Tensor:
        """Compute negative log likelihood loss."""
        precision = torch.exp(-logvar)
        loss = 0.5 * (logvar + (mean - target)**2 * precision)
        return torch.mean(loss)
    
    def save_model(self, path: str):
        """Save model state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path: str):
        """Load model state."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 