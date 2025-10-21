"""
Centralized Critic for CTDE (Centralized Training, Decentralized Execution)

This module implements a centralized value function that:
1. Takes global state as input (all jobs, machines, transbots)
2. Uses GNN to encode the complete workshop structure
3. Provides value estimates for all agents during training
4. Agents still use local observations for action selection (decentralized execution)

Key Benefits:
- More accurate value estimation using global information
- Reduced variance in policy gradients
- Better credit assignment across agents
- Single consistent baseline for all agents
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import numpy as np

from DFJSPT import dfjspt_params

# Conditional imports
if dfjspt_params.use_gnn_encoder:
    try:
        from DFJSPT.dfjspt_gnn_encoder import WorkshopGNN, TORCH_GEOMETRIC_AVAILABLE
        from torch_geometric.data import Data, Batch
        GNN_AVAILABLE = TORCH_GEOMETRIC_AVAILABLE
    except ImportError as e:
        print(f"Warning: Could not import GNN encoder for centralized critic: {e}")
        GNN_AVAILABLE = False
else:
    GNN_AVAILABLE = False


class GlobalStateAggregator(nn.Module):
    """
    Aggregates observations from all agents into a global state representation.
    
    This module collects:
    - All job features (from agent0)
    - All machine features (from agent1)
    - All transbot features (from agent2)
    - Additional global context (time, completion status, etc.)
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.n_jobs = config.get('n_jobs', dfjspt_params.n_jobs)
        self.n_machines = config.get('n_machines', dfjspt_params.n_machines)
        self.n_transbots = config.get('n_transbots', dfjspt_params.n_transbots)
        
        # Feature dimensions (will be set based on actual observations)
        self.job_feature_dim = None
        self.machine_feature_dim = None
        self.transbot_feature_dim = None
        
    def aggregate(self, obs_dict: Dict[str, Dict]) -> Dict[str, torch.Tensor]:
        """
        Aggregate observations from all agents into global state.
        
        Args:
            obs_dict: Dictionary with keys 'agent0', 'agent1', 'agent2'
                     Each value is a dict with 'observation' and 'action_mask'
                     
        Returns:
            global_state: Dictionary containing:
                - 'job_features': [batch_size, n_jobs, job_feature_dim]
                - 'machine_features': [batch_size, n_machines, machine_feature_dim]
                - 'transbot_features': [batch_size, n_transbots, transbot_feature_dim]
                - 'job_mask': [batch_size, n_jobs]
                - 'machine_mask': [batch_size, n_machines]
                - 'transbot_mask': [batch_size, n_transbots]
        """
        # Extract features from agent observations
        # Agent0 sees all jobs
        agent0_obs = obs_dict.get('agent0', {}).get('observation', None)
        agent0_mask = obs_dict.get('agent0', {}).get('action_mask', None)
        
        # Agent1 sees all machines
        agent1_obs = obs_dict.get('agent1', {}).get('observation', None)
        agent1_mask = obs_dict.get('agent1', {}).get('action_mask', None)
        
        # Agent2 sees all transbots
        agent2_obs = obs_dict.get('agent2', {}).get('observation', None)
        agent2_mask = obs_dict.get('agent2', {}).get('action_mask', None)
        
        # Convert to tensors if needed
        if agent0_obs is not None and not isinstance(agent0_obs, torch.Tensor):
            agent0_obs = torch.FloatTensor(agent0_obs)
        if agent1_obs is not None and not isinstance(agent1_obs, torch.Tensor):
            agent1_obs = torch.FloatTensor(agent1_obs)
        if agent2_obs is not None and not isinstance(agent2_obs, torch.Tensor):
            agent2_obs = torch.FloatTensor(agent2_obs)
            
        if agent0_mask is not None and not isinstance(agent0_mask, torch.Tensor):
            agent0_mask = torch.FloatTensor(agent0_mask)
        if agent1_mask is not None and not isinstance(agent1_mask, torch.Tensor):
            agent1_mask = torch.FloatTensor(agent1_mask)
        if agent2_mask is not None and not isinstance(agent2_mask, torch.Tensor):
            agent2_mask = torch.FloatTensor(agent2_mask)
        
        # Ensure batch dimension
        if agent0_obs is not None and len(agent0_obs.shape) == 2:
            agent0_obs = agent0_obs.unsqueeze(0)
        if agent1_obs is not None and len(agent1_obs.shape) == 2:
            agent1_obs = agent1_obs.unsqueeze(0)
        if agent2_obs is not None and len(agent2_obs.shape) == 2:
            agent2_obs = agent2_obs.unsqueeze(0)
            
        if agent0_mask is not None and len(agent0_mask.shape) == 1:
            agent0_mask = agent0_mask.unsqueeze(0)
        if agent1_mask is not None and len(agent1_mask.shape) == 1:
            agent1_mask = agent1_mask.unsqueeze(0)
        if agent2_mask is not None and len(agent2_mask.shape) == 1:
            agent2_mask = agent2_mask.unsqueeze(0)
        
        global_state = {
            'job_features': agent0_obs,
            'machine_features': agent1_obs,
            'transbot_features': agent2_obs,
            'job_mask': agent0_mask,
            'machine_mask': agent1_mask,
            'transbot_mask': agent2_mask,
        }
        
        return global_state


class CentralizedCriticNetwork(nn.Module):
    """
    Centralized value function network that uses global state.
    
    Two architecture options:
    1. GNN-based: Uses WorkshopGNN to encode global graph structure
    2. MLP-based: Concatenates all features and uses MLP
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.use_gnn = config.get('use_gnn', dfjspt_params.use_gnn_encoder) and GNN_AVAILABLE
        self.hidden_dim = config.get('hidden_dim', dfjspt_params.gnn_hidden_dim)
        
        self.n_jobs = config.get('n_jobs', dfjspt_params.n_jobs)
        self.n_machines = config.get('n_machines', dfjspt_params.n_machines)
        self.n_transbots = config.get('n_transbots', dfjspt_params.n_transbots)
        
        # Feature dimensions (from environment)
        self.job_feature_dim = config.get('job_feature_dim', 11)  # Will be updated
        self.machine_feature_dim = config.get('machine_feature_dim', 8)
        self.transbot_feature_dim = config.get('transbot_feature_dim', 9)
        
        if self.use_gnn:
            print("✅ Centralized Critic: Using GNN encoder")
            self._build_gnn_critic()
        else:
            print("ℹ️  Centralized Critic: Using MLP encoder")
            self._build_mlp_critic()
    
    def _build_gnn_critic(self):
        """Build GNN-based centralized critic"""
        # Node feature dimensions for complete workshop graph
        node_feature_dims = {
            'job': self.job_feature_dim,
            'machine': self.machine_feature_dim,
            'transbot': self.transbot_feature_dim,
            'operation': 0,  # Not used in centralized view
        }
        
        # Global GNN encoder
        self.gnn_encoder = WorkshopGNN(
            node_feature_dims=node_feature_dims,
            edge_feature_dim=4,
            hidden_dim=self.hidden_dim,
            num_layers=dfjspt_params.gnn_num_layers,
            gnn_type=dfjspt_params.gnn_type,
            pooling=dfjspt_params.gnn_pooling,
            dropout=dfjspt_params.gnn_dropout,
        )
        
        # Value head: maps global embedding to single value
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        
    def _build_mlp_critic(self):
        """Build MLP-based centralized critic"""
        # Calculate total feature dimension
        total_dim = (
            self.n_jobs * self.job_feature_dim +
            self.n_machines * self.machine_feature_dim +
            self.n_transbots * self.transbot_feature_dim
        )
        
        # MLP encoder
        self.mlp_encoder = nn.Sequential(
            nn.Linear(total_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )
    
    def forward(self, global_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through centralized critic.
        
        Args:
            global_state: Dictionary containing:
                - 'job_features': [batch_size, n_jobs, job_feature_dim]
                - 'machine_features': [batch_size, n_machines, machine_feature_dim]
                - 'transbot_features': [batch_size, n_transbots, transbot_feature_dim]
                - 'job_mask': [batch_size, n_jobs]
                - 'machine_mask': [batch_size, n_machines]
                - 'transbot_mask': [batch_size, n_transbots]
                
        Returns:
            values: [batch_size, 1] - Centralized value estimates
        """
        if self.use_gnn:
            return self._forward_gnn(global_state)
        else:
            return self._forward_mlp(global_state)
    
    def _forward_gnn(self, global_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """GNN forward pass"""
        # Convert global state to graph structure
        graph_data = self._build_global_graph(global_state)
        
        # Pass through GNN encoder
        gnn_output = self.gnn_encoder(
            node_features=graph_data['node_features'],
            node_types=graph_data['node_types'],
            edge_index=graph_data['edge_index'],
            edge_features=graph_data['edge_features'],
            batch=graph_data['batch'],
        )
        
        # Use global pooled embedding for value
        global_embedding = gnn_output['global_embedding']  # [batch_size, hidden_dim]
        
        # Compute value
        value = self.value_head(global_embedding)  # [batch_size, 1]
        
        return value
    
    def _forward_mlp(self, global_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """MLP forward pass"""
        # Flatten all features
        job_features = global_state['job_features']
        machine_features = global_state['machine_features']
        transbot_features = global_state['transbot_features']
        
        batch_size = job_features.shape[0]
        
        # Flatten each feature type
        job_flat = job_features.reshape(batch_size, -1)
        machine_flat = machine_features.reshape(batch_size, -1)
        transbot_flat = transbot_features.reshape(batch_size, -1)
        
        # Concatenate all features
        global_features = torch.cat([job_flat, machine_flat, transbot_flat], dim=1)
        
        # Pass through MLP encoder
        embedding = self.mlp_encoder(global_features)  # [batch_size, hidden_dim]
        
        # Compute value
        value = self.value_head(embedding)  # [batch_size, 1]
        
        return value
    
    def _build_global_graph(self, global_state: Dict[str, torch.Tensor]) -> Dict:
        """
        Build complete workshop graph from global state.
        
        Constructs a heterogeneous graph with:
        - Job nodes
        - Machine nodes
        - Transbot nodes
        - Edges representing relationships (job-machine compatibility, etc.)
        """
        job_features = global_state['job_features']
        machine_features = global_state['machine_features']
        transbot_features = global_state['transbot_features']
        
        batch_size = job_features.shape[0]
        
        # Stack all node features
        # Shape: [batch_size * (n_jobs + n_machines + n_transbots), max_feature_dim]
        max_feature_dim = max(
            job_features.shape[2],
            machine_features.shape[2],
            transbot_features.shape[2]
        )
        
        # Pad features to same dimension
        def pad_features(features, target_dim):
            current_dim = features.shape[2]
            if current_dim < target_dim:
                padding = torch.zeros(
                    features.shape[0], features.shape[1], target_dim - current_dim,
                    device=features.device
                )
                return torch.cat([features, padding], dim=2)
            return features
        
        job_features_padded = pad_features(job_features, max_feature_dim)
        machine_features_padded = pad_features(machine_features, max_feature_dim)
        transbot_features_padded = pad_features(transbot_features, max_feature_dim)
        
        # Flatten batch dimension
        job_features_flat = job_features_padded.reshape(-1, max_feature_dim)
        machine_features_flat = machine_features_padded.reshape(-1, max_feature_dim)
        transbot_features_flat = transbot_features_padded.reshape(-1, max_feature_dim)
        
        # Concatenate all node features
        node_features = torch.cat([
            job_features_flat,
            machine_features_flat,
            transbot_features_flat
        ], dim=0)
        
        # Node types
        n_total_jobs = batch_size * self.n_jobs
        n_total_machines = batch_size * self.n_machines
        n_total_transbots = batch_size * self.n_transbots
        
        node_types = torch.cat([
            torch.zeros(n_total_jobs, dtype=torch.long, device=job_features.device),  # 0 = job
            torch.ones(n_total_machines, dtype=torch.long, device=job_features.device),  # 1 = machine
            torch.full((n_total_transbots,), 2, dtype=torch.long, device=job_features.device),  # 2 = transbot
        ])
        
        # Build edges (simplified: fully connected within each batch)
        # In practice, you would use actual compatibility/precedence relationships
        edge_index = self._build_edge_index(batch_size)
        edge_features = torch.zeros(edge_index.shape[1], 4, device=job_features.device)
        
        # Batch indices for PyG
        batch = torch.cat([
            torch.arange(batch_size, device=job_features.device).repeat_interleave(self.n_jobs),
            torch.arange(batch_size, device=job_features.device).repeat_interleave(self.n_machines),
            torch.arange(batch_size, device=job_features.device).repeat_interleave(self.n_transbots),
        ])
        
        return {
            'node_features': node_features,
            'node_types': node_types,
            'edge_index': edge_index,
            'edge_features': edge_features,
            'batch': batch,
        }
    
    def _build_edge_index(self, batch_size: int) -> torch.Tensor:
        """
        Build edge connectivity for global graph.
        
        Creates edges:
        - Job -> Machine (potential assignments)
        - Machine -> Transbot (transport capability)
        - Job -> Job (precedence constraints)
        """
        edges = []
        
        nodes_per_batch = self.n_jobs + self.n_machines + self.n_transbots
        
        for b in range(batch_size):
            offset = b * nodes_per_batch
            job_offset = offset
            machine_offset = offset + self.n_jobs
            transbot_offset = offset + self.n_jobs + self.n_machines
            
            # Job -> Machine edges (all possible assignments)
            for j in range(self.n_jobs):
                for m in range(self.n_machines):
                    edges.append([job_offset + j, machine_offset + m])
            
            # Machine -> Transbot edges
            for m in range(self.n_machines):
                for t in range(self.n_transbots):
                    edges.append([machine_offset + m, transbot_offset + t])
            
            # Job -> Job edges (sequential processing)
            for j in range(self.n_jobs - 1):
                edges.append([job_offset + j, job_offset + j + 1])
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
        else:
            # Empty edge index if no edges
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        return edge_index


class CentralizedCritic:
    """
    Main interface for centralized critic.
    
    Usage:
        critic = CentralizedCritic(config)
        value = critic.compute_value(obs_dict)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = {}
        
        self.config = config
        self.aggregator = GlobalStateAggregator(config)
        self.network = CentralizedCriticNetwork(config)
        
        # Move to GPU if available
        if torch.cuda.is_available() and dfjspt_params.num_gpu > 0:
            self.network = self.network.cuda()
            print("✅ Centralized Critic moved to GPU")
    
    def compute_value(self, obs_dict: Dict[str, Dict]) -> torch.Tensor:
        """
        Compute centralized value estimate.
        
        Args:
            obs_dict: Dictionary with observations from all agents
            
        Returns:
            value: Centralized value estimate [batch_size, 1]
        """
        # Aggregate global state
        global_state = self.aggregator.aggregate(obs_dict)
        
        # Move to same device as network
        device = next(self.network.parameters()).device
        global_state = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                       for k, v in global_state.items()}
        
        # Compute value
        with torch.no_grad():
            value = self.network(global_state)
        
        return value
    
    def compute_value_for_training(self, obs_dict: Dict[str, Dict]) -> torch.Tensor:
        """
        Compute value with gradients enabled (for training).
        
        Args:
            obs_dict: Dictionary with observations from all agents
            
        Returns:
            value: Centralized value estimate [batch_size, 1]
        """
        # Aggregate global state
        global_state = self.aggregator.aggregate(obs_dict)
        
        # Move to same device as network
        device = next(self.network.parameters()).device
        global_state = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                       for k, v in global_state.items()}
        
        # Compute value (with gradients)
        value = self.network(global_state)
        
        return value
    
    def parameters(self):
        """Return network parameters for optimizer"""
        return self.network.parameters()
    
    def state_dict(self):
        """Return network state dict for saving"""
        return self.network.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load network state dict"""
        self.network.load_state_dict(state_dict)


# Helper function for creating centralized critic
def create_centralized_critic(config: Optional[Dict] = None) -> CentralizedCritic:
    """
    Factory function to create centralized critic.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        CentralizedCritic instance
    """
    if config is None:
        config = {
            'use_gnn': dfjspt_params.use_gnn_encoder,
            'hidden_dim': dfjspt_params.gnn_hidden_dim,
            'n_jobs': dfjspt_params.n_jobs,
            'n_machines': dfjspt_params.n_machines,
            'n_transbots': dfjspt_params.n_transbots,
        }
    
    return CentralizedCritic(config)


if __name__ == "__main__":
    """Test centralized critic"""
    print("="*80)
    print("Testing Centralized Critic")
    print("="*80)
    
    # Create config
    config = {
        'use_gnn': True,
        'hidden_dim': 128,
        'n_jobs': 20,
        'n_machines': 5,
        'n_transbots': 3,
        'job_feature_dim': 11,
        'machine_feature_dim': 8,
        'transbot_feature_dim': 9,
    }
    
    # Create critic
    critic = create_centralized_critic(config)
    
    # Create dummy observations
    batch_size = 4
    obs_dict = {
        'agent0': {
            'observation': torch.randn(batch_size, 20, 11),
            'action_mask': torch.ones(batch_size, 20),
        },
        'agent1': {
            'observation': torch.randn(batch_size, 5, 8),
            'action_mask': torch.ones(batch_size, 5),
        },
        'agent2': {
            'observation': torch.randn(batch_size, 3, 9),
            'action_mask': torch.ones(batch_size, 3),
        },
    }
    
    # Compute value
    print("\nComputing centralized value...")
    value = critic.compute_value(obs_dict)
    print(f"✅ Value shape: {value.shape}")
    print(f"   Value range: [{value.min():.4f}, {value.max():.4f}]")
    
    # Test with training mode
    print("\nComputing value for training (with gradients)...")
    value_train = critic.compute_value_for_training(obs_dict)
    print(f"✅ Value shape: {value_train.shape}")
    print(f"   Requires grad: {value_train.requires_grad}")
    
    # Count parameters
    n_params = sum(p.numel() for p in critic.parameters())
    print(f"\n✅ Total parameters: {n_params:,}")
    
    print("\n" + "="*80)
    print("Centralized Critic test passed!")
    print("="*80)
