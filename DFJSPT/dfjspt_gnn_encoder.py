"""
Graph Neural Network Encoder for DFJSP-T Environment

Implements a unified GNN that encodes the dynamic workshop state graph into
embedding vectors for all three agents (strategy, job, machine, transbot).

Graph Structure:
- Nodes: Jobs, Machines, Transbots, Operations
- Edges: Job-Operation, Operation-Machine (capability), Transbot-Location

Node Features:
- Job: [remaining_ops, priority, slack_time, ...]
- Machine: [utilization, queue_length, capability_vector, ...]
- Transbot: [location, carrying_status, battery, ...]
- Operation: [processing_time, remaining_time, status, ...]

Edge Features:
- Job-Op: [operation_index, precedence]
- Op-Machine: [processing_time, capability]
- Transbot-Location: [distance, travel_time]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

try:
    import torch_geometric
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: torch_geometric not installed. GNN encoder will not be available.")
    print("Install with: pip install torch-geometric")


class GraphAttentionPooling(nn.Module):
    """
    Attention-based graph pooling for aggregating node embeddings.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention_weights = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x, batch=None):
        """
        Args:
            x: Node features [num_nodes, hidden_dim]
            batch: Batch assignment for each node [num_nodes]
        Returns:
            pooled: [batch_size, hidden_dim]
        """
        # Compute attention scores
        attn_scores = self.attention_weights(x)  # [num_nodes, 1]
        
        if batch is None:
            # Single graph
            attn_weights = F.softmax(attn_scores, dim=0)
            pooled = torch.sum(x * attn_weights, dim=0, keepdim=True)
        else:
            # Multiple graphs in batch
            attn_weights = torch_geometric.utils.softmax(attn_scores, batch, dim=0)
            pooled = torch_geometric.nn.global_add_pool(x * attn_weights, batch)
        
        return pooled


class WorkshopGNN(nn.Module):
    """
    Graph Neural Network for encoding workshop state.
    
    Architecture:
    1. Node feature embedding
    2. Multiple GNN layers (GCN/GAT/GraphSAGE)
    3. Pooling for graph-level representation
    4. Agent-specific heads (small MLPs or attention pooling)
    """
    
    def __init__(
        self,
        node_feature_dims: Dict[str, int],
        edge_feature_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 3,
        gnn_type: str = "gat",  # "gcn", "gat", "sage"
        pooling: str = "attention",  # "mean", "max", "attention"
        dropout: float = 0.1,
    ):
        super().__init__()
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric is required for GNN encoder")
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.pooling = pooling
        
        # Node type embeddings (to distinguish job/machine/transbot/operation nodes)
        self.node_type_embedding = nn.Embedding(4, hidden_dim // 4)  # 4 types
        
        # Feature projection for different node types
        self.node_encoders = nn.ModuleDict()
        for node_type, feat_dim in node_feature_dims.items():
            self.node_encoders[node_type] = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim - hidden_dim // 4),
                nn.LayerNorm(hidden_dim - hidden_dim // 4),
                nn.ReLU(),
            )
        
        # Edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
        ) if edge_feature_dim > 0 else None
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            if gnn_type == "gcn":
                self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
            elif gnn_type == "gat":
                self.gnn_layers.append(GATConv(hidden_dim, hidden_dim // 4, heads=4, dropout=dropout))
            elif gnn_type == "sage":
                self.gnn_layers.append(SAGEConv(hidden_dim, hidden_dim))
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")
        
        # Layer normalization and dropout
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        
        # Graph-level pooling
        if pooling == "attention":
            self.pooling_layer = GraphAttentionPooling(hidden_dim)
        elif pooling == "mean":
            self.pooling_layer = global_mean_pool
        elif pooling == "max":
            self.pooling_layer = global_max_pool
        else:
            raise ValueError(f"Unknown pooling type: {pooling}")
        
        # Agent-specific heads (small MLPs)
        self.strategy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.job_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.machine_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.transbot_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(
        self,
        node_features: Dict[str, torch.Tensor],
        node_types: torch.LongTensor,
        edge_index: torch.LongTensor,
        edge_features: Optional[torch.Tensor] = None,
        batch: Optional[torch.LongTensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through GNN.
        
        Args:
            node_features: Dict of {node_type: features} for each node type
            node_types: Tensor of node type IDs [num_nodes]
            edge_index: Edge connectivity [2, num_edges]
            edge_features: Edge features [num_edges, edge_feat_dim]
            batch: Batch assignment for nodes [num_nodes] (for batched graphs)
        
        Returns:
            Dict with keys:
                - 'strategy_embedding': [batch_size, hidden_dim]
                - 'job_embedding': [batch_size, hidden_dim]
                - 'machine_embedding': [batch_size, hidden_dim]
                - 'transbot_embedding': [batch_size, hidden_dim]
                - 'node_embeddings': [num_nodes, hidden_dim]
        """
        # Encode node features based on type
        # Count nodes of each type and create index mapping
        type_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # job, machine, transbot, operation
        type_indices = {0: [], 1: [], 2: [], 3: []}
        
        for i in range(node_types.size(0)):
            type_id = node_types[i].item()
            type_indices[type_id].append(i)
            type_counts[type_id] += 1
        
        # Pre-allocate tensor for all node embeddings
        num_nodes = node_types.size(0)
        x = torch.zeros(num_nodes, self.hidden_dim, device=node_types.device)
        
        # Process each node type separately
        type_names = ['job', 'machine', 'transbot', 'operation']
        for type_id, type_name in enumerate(type_names):
            if type_counts[type_id] == 0:
                continue
            
            indices = type_indices[type_id]
            if type_name in node_features:
                # Get features for this type
                feat = node_features[type_name]  # [num_nodes_of_type, feat_dim]
                encoded_feat = self.node_encoders[type_name](feat)  # [num_nodes_of_type, hidden_dim - hidden_dim//4]
                
                # Get type embeddings
                type_ids = torch.full((len(indices),), type_id, dtype=torch.long, device=node_types.device)
                type_emb = self.node_type_embedding(type_ids)  # [num_nodes_of_type, hidden_dim//4]
                
                # Concatenate
                node_emb = torch.cat([encoded_feat, type_emb], dim=-1)  # [num_nodes_of_type, hidden_dim]
                
                # Assign to correct positions
                for local_idx, global_idx in enumerate(indices):
                    x[global_idx] = node_emb[local_idx]
        
        # Encode edge features if provided
        edge_attr = None
        if self.edge_encoder is not None and edge_features is not None:
            edge_attr = self.edge_encoder(edge_features)
        
        # Apply GNN layers
        for i, gnn_layer in enumerate(self.gnn_layers):
            x_residual = x
            
            # GNN convolution
            if self.gnn_type in ["gcn", "sage"]:
                x = gnn_layer(x, edge_index)
            elif self.gnn_type == "gat":
                x = gnn_layer(x, edge_index)
            
            # Layer normalization and residual connection
            x = self.layer_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
            
            # Residual connection
            if i > 0:  # Skip first layer residual
                x = x + x_residual
        
        node_embeddings = x  # [num_nodes, hidden_dim]
        
        # Graph-level pooling
        if self.pooling == "attention":
            graph_embedding = self.pooling_layer(x, batch)
        else:
            graph_embedding = self.pooling_layer(x, batch)
        
        # Agent-specific embeddings
        strategy_emb = self.strategy_head(graph_embedding)
        job_emb = self.job_head(graph_embedding)
        machine_emb = self.machine_head(graph_embedding)
        transbot_emb = self.transbot_head(graph_embedding)
        
        return {
            'strategy_embedding': strategy_emb,
            'job_embedding': job_emb,
            'machine_embedding': machine_emb,
            'transbot_embedding': transbot_emb,
            'node_embeddings': node_embeddings,
            'graph_embedding': graph_embedding,
        }


class WorkshopGraphBuilder:
    """
    Utility class to build PyTorch Geometric graph from DFJSP-T environment state.
    """
    
    @staticmethod
    def build_graph_from_env_state(env_state: Dict) -> Data:
        """
        Convert environment state to PyTorch Geometric Data object.
        
        Args:
            env_state: Dictionary containing:
                - jobs: List of job states
                - machines: List of machine states
                - transbots: List of transbot states
                - operations: List of operation states
                - job_op_edges: List of (job_id, op_id) edges
                - op_machine_edges: List of (op_id, machine_id, processing_time) edges
                - transbot_location_edges: List of (transbot_id, location_id) edges
        
        Returns:
            graph: PyTorch Geometric Data object
        """
        # Node features extraction
        job_features = torch.FloatTensor(env_state.get('job_features', []))
        machine_features = torch.FloatTensor(env_state.get('machine_features', []))
        transbot_features = torch.FloatTensor(env_state.get('transbot_features', []))
        operation_features = torch.FloatTensor(env_state.get('operation_features', []))
        
        # Node type IDs: 0=job, 1=machine, 2=transbot, 3=operation
        num_jobs = len(job_features)
        num_machines = len(machine_features)
        num_transbots = len(transbot_features)
        num_operations = len(operation_features)
        
        node_types = torch.cat([
            torch.zeros(num_jobs, dtype=torch.long),
            torch.ones(num_machines, dtype=torch.long),
            torch.full((num_transbots,), 2, dtype=torch.long),
            torch.full((num_operations,), 3, dtype=torch.long),
        ])
        
        # Build node features dict
        node_features = {
            'job': job_features,
            'machine': machine_features,
            'transbot': transbot_features,
            'operation': operation_features,
        }
        
        # Build edge index
        edge_list = []
        edge_features_list = []
        
        # Job-Operation edges
        for job_id, op_id in env_state.get('job_op_edges', []):
            edge_list.append([job_id, num_jobs + num_machines + num_transbots + op_id])
            edge_list.append([num_jobs + num_machines + num_transbots + op_id, job_id])  # Bidirectional
            edge_features_list.append([1.0, 0.0, 0.0, 0.0])  # Edge type: job-op
            edge_features_list.append([1.0, 0.0, 0.0, 0.0])
        
        # Operation-Machine edges
        for op_id, machine_id, proc_time in env_state.get('op_machine_edges', []):
            op_node_id = num_jobs + num_machines + num_transbots + op_id
            machine_node_id = num_jobs + machine_id
            edge_list.append([op_node_id, machine_node_id])
            edge_list.append([machine_node_id, op_node_id])  # Bidirectional
            edge_features_list.append([0.0, 1.0, proc_time, 0.0])  # Edge type: op-machine
            edge_features_list.append([0.0, 1.0, proc_time, 0.0])
        
        # Transbot-Location edges (simplified)
        for transbot_id, location_id in env_state.get('transbot_location_edges', []):
            transbot_node_id = num_jobs + num_machines + transbot_id
            # Connect to machine at location
            if location_id < num_machines:
                machine_node_id = num_jobs + location_id
                edge_list.append([transbot_node_id, machine_node_id])
                edge_list.append([machine_node_id, transbot_node_id])
                edge_features_list.append([0.0, 0.0, 0.0, 1.0])  # Edge type: transbot-location
                edge_features_list.append([0.0, 0.0, 0.0, 1.0])
        
        if len(edge_list) > 0:
            edge_index = torch.LongTensor(edge_list).t().contiguous()
            edge_features = torch.FloatTensor(edge_features_list)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_features = torch.zeros((0, 4), dtype=torch.float)
        
        # Create PyTorch Geometric Data object
        graph = Data(
            x=node_features,  # Store as dict for now
            node_types=node_types,
            edge_index=edge_index,
            edge_attr=edge_features,
            num_nodes=num_jobs + num_machines + num_transbots + num_operations,
        )
        
        return graph


# Example usage and testing
if __name__ == "__main__":
    print("Testing WorkshopGNN...")
    
    if not TORCH_GEOMETRIC_AVAILABLE:
        print("torch_geometric not available. Skipping tests.")
    else:
        # Define node feature dimensions
        node_feature_dims = {
            'job': 10,
            'machine': 8,
            'transbot': 6,
            'operation': 12,
        }
        
        # Create GNN model
        gnn = WorkshopGNN(
            node_feature_dims=node_feature_dims,
            hidden_dim=128,
            num_layers=3,
            gnn_type="gat",
            pooling="attention",
        )
        
        print(f"GNN model created with {sum(p.numel() for p in gnn.parameters())} parameters")
        
        # Create dummy graph
        num_jobs = 5
        num_machines = 3
        num_transbots = 2
        num_operations = 10
        num_nodes = num_jobs + num_machines + num_transbots + num_operations
        
        node_features = {
            'job': torch.randn(num_jobs, 10),
            'machine': torch.randn(num_machines, 8),
            'transbot': torch.randn(num_transbots, 6),
            'operation': torch.randn(num_operations, 12),
        }
        
        node_types = torch.cat([
            torch.zeros(num_jobs, dtype=torch.long),
            torch.ones(num_machines, dtype=torch.long),
            torch.full((num_transbots,), 2, dtype=torch.long),
            torch.full((num_operations,), 3, dtype=torch.long),
        ])
        
        # Random edges
        edge_index = torch.randint(0, num_nodes, (2, 50))
        edge_features = torch.randn(50, 4)
        
        # Forward pass
        output = gnn(node_features, node_types, edge_index, edge_features)
        
        print("\nOutput shapes:")
        for key, tensor in output.items():
            if 'embedding' in key:
                print(f"  {key}: {tensor.shape}")
        
        print("\n✅ GNN encoder test passed!")
