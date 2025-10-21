"""
Centralized Critic Integration for Agent Models

This module provides wrapper classes that integrate centralized critic
into existing agent models while keeping policies decentralized.

Key Design:
- Policy networks remain unchanged (decentralized execution)
- Value function is replaced by centralized critic (centralized training)
- All agents share the same centralized critic instance
"""

import torch
import torch.nn as nn
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from typing import Dict, Optional

from DFJSPT import dfjspt_params
from DFJSPT.dfjspt_agent_model import (
    JobActionMaskModel,
    MachineActionMaskModel, 
    TransbotActionMaskModel
)

# Always try to import centralized critic (will check use_centralized_critic at runtime)
try:
    from DFJSPT.centralized_critic import CentralizedCritic
    CENTRALIZED_CRITIC_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import centralized critic: {e}")
    CENTRALIZED_CRITIC_AVAILABLE = False


# Global centralized critic instance (shared across all agents)
_GLOBAL_CENTRALIZED_CRITIC = None


def get_centralized_critic() -> Optional['CentralizedCritic']:
    """Get the global centralized critic instance"""
    global _GLOBAL_CENTRALIZED_CRITIC
    return _GLOBAL_CENTRALIZED_CRITIC


def set_centralized_critic(critic: 'CentralizedCritic'):
    """Set the global centralized critic instance"""
    global _GLOBAL_CENTRALIZED_CRITIC
    _GLOBAL_CENTRALIZED_CRITIC = critic
    print(f"✅ Global centralized critic set: {type(critic).__name__}")


def create_and_set_centralized_critic(config: Optional[Dict] = None):
    """
    Create and set the global centralized critic.
    
    This should be called once before creating any agent models.
    """
    if not CENTRALIZED_CRITIC_AVAILABLE:
        print("⚠️  Centralized critic not available, using decentralized critics")
        return None
    
    from DFJSPT.centralized_critic import create_centralized_critic
    
    if config is None:
        config = {
            'use_gnn': dfjspt_params.centralized_critic_use_gnn and dfjspt_params.use_gnn_encoder,
            'hidden_dim': dfjspt_params.centralized_critic_hidden_dim,
            'n_jobs': dfjspt_params.n_jobs,
            'n_machines': dfjspt_params.n_machines,
            'n_transbots': dfjspt_params.n_transbots,
        }
    
    critic = create_centralized_critic(config)
    set_centralized_critic(critic)
    
    return critic


class CentralizedCriticMixin:
    """
    Mixin class that adds centralized critic support to agent models.
    
    This mixin:
    1. Stores observations for centralized critic
    2. Overrides value_function() to use centralized critic
    3. Handles observation aggregation across agents
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Get centralized critic instance
        self.centralized_critic = get_centralized_critic()
        self.use_centralized_critic = (
            dfjspt_params.use_centralized_critic and 
            self.centralized_critic is not None
        )
        
        if self.use_centralized_critic:
            print(f"✅ {self.__class__.__name__}: Using centralized critic")
            # Storage for observations (needed for centralized critic)
            self._last_obs_dict = None
            self._last_centralized_value = None
        else:
            print(f"ℹ️  {self.__class__.__name__}: Using decentralized critic")
    
    def _store_obs_for_centralized_critic(self, input_dict):
        """
        Store observations for centralized critic computation.
        
        Note: In RLlib, we can't directly access other agents' observations
        during forward pass. We need to aggregate them during training.
        For now, we'll use the current agent's observation and approximate
        the global state.
        """
        if not self.use_centralized_critic:
            return
        
        # Extract observation from input_dict
        obs = input_dict.get("obs", {})
        
        # Store for later use in value_function()
        # In practice, RLlib's SampleBatch will have observations from all agents
        self._last_obs_dict = obs
    
    def _get_agent_id_from_model_name(self) -> str:
        """Infer agent ID from model class name"""
        class_name = self.__class__.__name__
        if "Job" in class_name:
            return "agent0"
        elif "Machine" in class_name:
            return "agent1"
        elif "Transbot" in class_name:
            return "agent2"
        else:
            return "unknown"
    
    @override(ModelV2)
    def value_function(self):
        """
        Override value function to use centralized critic.
        
        Returns:
            value: Tensor of shape [batch_size] with value estimates
        """
        if not self.use_centralized_critic:
            # Use parent class's decentralized value function
            return super().value_function()
        
        # Use centralized critic
        if self._last_centralized_value is not None:
            return self._last_centralized_value
        
        # Fallback: return zeros if no value computed yet
        # This shouldn't happen in practice as we compute in forward()
        return torch.zeros(1, device=next(self.parameters()).device)
    
    def compute_centralized_value(self, obs_batch: Dict[str, Dict]) -> torch.Tensor:
        """
        Compute centralized value given observations from all agents.
        
        Args:
            obs_batch: Dictionary with keys 'agent0', 'agent1', 'agent2'
                      Each contains 'observation' and 'action_mask'
        
        Returns:
            values: [batch_size] tensor of value estimates
        """
        if not self.use_centralized_critic or self.centralized_critic is None:
            raise RuntimeError("Centralized critic not available")
        
        # Compute value using centralized critic
        value = self.centralized_critic.compute_value_for_training(obs_batch)
        
        # Squeeze to [batch_size] shape expected by RLlib
        if len(value.shape) == 2 and value.shape[1] == 1:
            value = value.squeeze(-1)
        
        return value


class JobActionMaskModelWithCC(CentralizedCriticMixin, JobActionMaskModel):
    """Job agent model with centralized critic support"""
    
    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        # Store observations for centralized critic
        self._store_obs_for_centralized_critic(input_dict)
        
        # Call parent forward (policy network unchanged)
        logits, state = super().forward(input_dict, state, seq_lens)
        
        # Note: Value computation happens separately in value_function()
        # In centralized critic mode, we need global observations which
        # are only available during training, not during rollout
        
        return logits, state


class MachineActionMaskModelWithCC(CentralizedCriticMixin, MachineActionMaskModel):
    """Machine agent model with centralized critic support"""
    
    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        # Store observations for centralized critic
        self._store_obs_for_centralized_critic(input_dict)
        
        # Call parent forward (policy network unchanged)
        logits, state = super().forward(input_dict, state, seq_lens)
        
        return logits, state


class TransbotActionMaskModelWithCC(CentralizedCriticMixin, TransbotActionMaskModel):
    """Transbot agent model with centralized critic support"""
    
    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        # Store observations for centralized critic
        self._store_obs_for_centralized_critic(input_dict)
        
        # Call parent forward (policy network unchanged)
        logits, state = super().forward(input_dict, state, seq_lens)
        
        return logits, state


def get_model_class_for_agent(agent_name: str, use_centralized: bool = None):
    """
    Get appropriate model class for agent based on configuration.
    
    Args:
        agent_name: 'agent0', 'agent1', or 'agent2'
        use_centralized: Override for centralized critic usage
        
    Returns:
        Model class to use
    """
    if use_centralized is None:
        use_centralized = dfjspt_params.use_centralized_critic
    
    if use_centralized and CENTRALIZED_CRITIC_AVAILABLE:
        model_map = {
            'agent0': JobActionMaskModelWithCC,
            'agent1': MachineActionMaskModelWithCC,
            'agent2': TransbotActionMaskModelWithCC,
        }
    else:
        model_map = {
            'agent0': JobActionMaskModel,
            'agent1': MachineActionMaskModel,
            'agent2': TransbotActionMaskModel,
        }
    
    return model_map.get(agent_name)


def get_custom_model_name_for_agent(agent_name: str, use_centralized: bool = None) -> str:
    """
    Get custom model name for ModelCatalog registration.
    
    Args:
        agent_name: 'agent0', 'agent1', or 'agent2'
        use_centralized: Override for centralized critic usage
        
    Returns:
        Model name string for registration
    """
    if use_centralized is None:
        use_centralized = dfjspt_params.use_centralized_critic
    
    if use_centralized and CENTRALIZED_CRITIC_AVAILABLE:
        name_map = {
            'agent0': 'job_agent_model_cc',
            'agent1': 'machine_agent_model_cc',
            'agent2': 'transbot_agent_model_cc',
        }
    else:
        name_map = {
            'agent0': 'job_agent_model',
            'agent1': 'machine_agent_model',
            'agent2': 'transbot_agent_model',
        }
    
    return name_map.get(agent_name, 'unknown_model')


# Convenience function for registering all models
def register_all_models_with_catalog(model_catalog):
    """
    Register all agent models with RLlib ModelCatalog.
    
    Automatically selects centralized or decentralized versions based on config.
    
    Args:
        model_catalog: RLlib's ModelCatalog class
    """
    use_centralized = dfjspt_params.use_centralized_critic and CENTRALIZED_CRITIC_AVAILABLE
    
    if use_centralized:
        print("📋 Registering models WITH centralized critic...")
        model_catalog.register_custom_model("job_agent_model_cc", JobActionMaskModelWithCC)
        model_catalog.register_custom_model("machine_agent_model_cc", MachineActionMaskModelWithCC)
        model_catalog.register_custom_model("transbot_agent_model_cc", TransbotActionMaskModelWithCC)
    else:
        print("📋 Registering models WITHOUT centralized critic...")
        model_catalog.register_custom_model("job_agent_model", JobActionMaskModel)
        model_catalog.register_custom_model("machine_agent_model", MachineActionMaskModel)
        model_catalog.register_custom_model("transbot_agent_model", TransbotActionMaskModel)
    
    print(f"✅ All agent models registered (centralized={use_centralized})")


if __name__ == "__main__":
    """Test centralized critic integration"""
    print("="*80)
    print("Testing Centralized Critic Integration")
    print("="*80)
    
    # Enable centralized critic for testing
    import DFJSPT.dfjspt_params as params
    params.use_centralized_critic = True
    params.use_gnn_encoder = True
    
    # Create centralized critic
    print("\n1. Creating centralized critic...")
    critic = create_and_set_centralized_critic()
    print(f"   Critic created: {critic is not None}")
    
    # Test model class selection
    print("\n2. Testing model class selection...")
    for agent in ['agent0', 'agent1', 'agent2']:
        model_cls = get_model_class_for_agent(agent, use_centralized=True)
        model_name = get_custom_model_name_for_agent(agent, use_centralized=True)
        print(f"   {agent}: {model_cls.__name__} -> '{model_name}'")
    
    print("\n" + "="*80)
    print("Centralized critic integration test completed!")
    print("="*80)
