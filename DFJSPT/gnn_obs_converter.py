"""
Helper functions for converting DFJSP-T observations to GNN graphs.

This module provides utilities to convert RLlib observations into
PyTorch Geometric graph structures for the GNN encoder.
"""

import torch
import numpy as np
from typing import Dict, Tuple, List

try:
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False


def observation_to_graph(observation: np.ndarray, agent_type: str, device='cpu') -> Dict:
    """
    Convert flat observation array to graph structure for GNN.
    
    Args:
        observation: Numpy array from environment [n_entities, n_features]
        agent_type: One of "job", "machine", "transbot"
        device: PyTorch device
    
    Returns:
        Dict with graph components ready for GNN
    """
    obs_tensor = torch.FloatTensor(observation).to(device)
    n_entities, n_features = obs_tensor.shape
    
    # Build simple graph structure
    # For now, create a fully connected graph within entity type
    # In future, can add cross-type edges
    
    edge_list = []
    for i in range(n_entities):
        for j in range(n_entities):
            if i != j:
                edge_list.append([i, j])
    
    if len(edge_list) > 0:
        edge_index = torch.LongTensor(edge_list).t().contiguous().to(device)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
    
    # Node features
    node_features = {
        agent_type: obs_tensor
    }
    
    # All nodes are of the same type
    node_types = torch.zeros(n_entities, dtype=torch.long, device=device)
    if agent_type == 'machine':
        node_types.fill_(1)
    elif agent_type == 'transbot':
        node_types.fill_(2)
    # job is 0, which is already filled
    
    return {
        'node_features': node_features,
        'node_types': node_types,
        'edge_index': edge_index,
        'edge_features': None,
        'batch': None,
    }


def batch_observations_to_graphs(observations: torch.Tensor, agent_type: str) -> Dict:
    """
    Convert batched observations to batched graphs.
    
    Args:
        observations: [batch_size, n_entities, n_features]
        agent_type: One of "job", "machine", "transbot"
    
    Returns:
        Dict with batched graph components
    """
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise ImportError("torch_geometric required for GNN encoder")
    
    batch_size = observations.shape[0]
    device = observations.device
    
    # Build individual graphs
    graphs = []
    for i in range(batch_size):
        graph_dict = observation_to_graph(
            observations[i].cpu().numpy(),
            agent_type=agent_type,
            device=device
        )
        graphs.append(graph_dict)
    
    # Merge into batched graph
    all_node_features = {agent_type: []}
    all_node_types = []
    all_edge_indices = []
    batch_assignment = []
    
    node_offset = 0
    for batch_idx, graph in enumerate(graphs):
        # Node features
        node_feat = graph['node_features'][agent_type]
        all_node_features[agent_type].append(node_feat)
        
        # Node types
        all_node_types.append(graph['node_types'])
        
        # Edge index (with offset)
        edge_index = graph['edge_index'] + node_offset
        all_edge_indices.append(edge_index)
        
        # Batch assignment
        n_nodes = node_feat.shape[0]
        batch_assignment.extend([batch_idx] * n_nodes)
        
        node_offset += n_nodes
    
    # Concatenate
    merged_node_features = {
        agent_type: torch.cat(all_node_features[agent_type], dim=0)
    }
    merged_node_types = torch.cat(all_node_types, dim=0)
    merged_edge_index = torch.cat(all_edge_indices, dim=1)
    merged_batch = torch.LongTensor(batch_assignment).to(device)
    
    return {
        'node_features': merged_node_features,
        'node_types': merged_node_types,
        'edge_index': merged_edge_index,
        'edge_features': None,
        'batch': merged_batch,
    }
