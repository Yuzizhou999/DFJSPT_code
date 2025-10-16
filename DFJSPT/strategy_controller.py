"""
Strategy Controller for Hierarchical DFJSP-T Training

This implements the strategy layer as a meta-controller outside RLlib's agent framework,
avoiding the "single trajectory" error while maintaining hierarchical decision-making.

Architecture:
- Strategy network learns to set preference vectors
- Operates at episode granularity (not step-by-step)
- Updates based on episode outcomes
- Controls tactical policies through environment preference setting
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import pickle
import os


class StrategyNetwork(nn.Module):
    """
    Neural network for strategy layer.
    
    Input: Episode context (state statistics, historical performance)
    Output: Preference vector [efficiency_weight, cost_weight, delivery_weight]
    """
    
    def __init__(self, obs_dim=19, action_dim=3, hidden_dim=128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        
    def forward(self, x):
        """
        Output raw logits, will apply softmax externally
        """
        return self.network(x)
    
    def get_preference(self, obs):
        """
        Get preference vector from observation
        
        Args:
            obs: Strategy observation (numpy array or tensor)
            
        Returns:
            preference_vector: Softmax-normalized weights [efficiency, cost, delivery]
        """
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs)
        
        with torch.no_grad():
            logits = self.forward(obs)
            # Apply softmax to get valid probability distribution
            preference = torch.softmax(logits, dim=-1)
            
        return preference.numpy()


class StrategyController:
    """
    Meta-controller for strategy layer learning.
    
    Uses Reinforcement Learning to learn preference setting policy:
    - Observation: Episode context features
    - Action: Preference vector for multi-objective scalarization
    - Reward: Episode performance (customizable objective)
    """
    
    def __init__(
        self,
        obs_dim=19,
        action_dim=3,
        hidden_dim=128,
        learning_rate=3e-4,
        gamma=0.99,
        buffer_size=1000,
    ):
        self.network = StrategyNetwork(obs_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.gamma = gamma
        
        # Experience buffer for strategy learning
        self.buffer = deque(maxlen=buffer_size)
        
        # Statistics tracking
        self.episode_count = 0
        self.total_updates = 0
        
    def select_preference(self, context_obs, exploration=True, epsilon=0.1):
        """
        Select preference vector for upcoming episode
        
        Args:
            context_obs: Context observation for strategy decision
            exploration: Whether to add exploration noise
            epsilon: Exploration noise level
            
        Returns:
            preference_vector: [efficiency_weight, cost_weight, delivery_weight]
        """
        preference = self.network.get_preference(context_obs)
        
        if exploration and np.random.random() < epsilon:
            # Exploration: sample random preference with Dirichlet distribution
            # This ensures valid probability distribution
            preference = np.random.dirichlet(np.ones(3))
        
        return preference
    
    def store_experience(self, context_obs, preference, reward, next_context_obs, done):
        """
        Store episode-level experience
        
        Args:
            context_obs: Initial context observation
            preference: Selected preference vector
            reward: Episode return (can be multi-objective)
            next_context_obs: Context after episode
            done: Whether this ends a meta-episode
        """
        self.buffer.append({
            'context_obs': context_obs,
            'preference': preference,
            'reward': reward,
            'next_context_obs': next_context_obs,
            'done': done,
        })
    
    def update(self, batch_size=32):
        """
        Update strategy network using collected experiences
        
        Uses policy gradient / actor-critic style update
        For simplicity, we use supervised learning on successful preferences
        """
        if len(self.buffer) < batch_size:
            return None
        
        # Sample batch
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        # Prepare tensors
        context_obs = torch.FloatTensor([exp['context_obs'] for exp in batch])
        preferences = torch.FloatTensor([exp['preference'] for exp in batch])
        rewards = torch.FloatTensor([exp['reward'] for exp in batch])
        
        # Normalize rewards for stable learning
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Forward pass
        logits = self.network(context_obs)
        
        # Compute log probabilities of selected preferences
        # We treat this as a categorical distribution over preference combinations
        # For continuous preference, use cross-entropy between predicted and target
        pred_prefs = torch.softmax(logits, dim=-1)
        
        # Loss: minimize KL divergence weighted by rewards (policy gradient style)
        # Higher reward -> stronger pull toward this preference
        kl_loss = -torch.sum(preferences * torch.log(pred_prefs + 1e-8), dim=-1)
        weighted_loss = (kl_loss * rewards).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        
        self.total_updates += 1
        
        return weighted_loss.item()
    
    def save(self, path):
        """Save strategy controller state"""
        state = {
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episode_count': self.episode_count,
            'total_updates': self.total_updates,
        }
        torch.save(state, path)
        print(f"Strategy controller saved to {path}")
    
    def load(self, path):
        """Load strategy controller state"""
        if not os.path.exists(path):
            print(f"Warning: Strategy checkpoint not found at {path}")
            return
        
        state = torch.load(path)
        self.network.load_state_dict(state['network'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.episode_count = state.get('episode_count', 0)
        self.total_updates = state.get('total_updates', 0)
        print(f"Strategy controller loaded from {path}")
        print(f"  Episodes: {self.episode_count}, Updates: {self.total_updates}")


class EpisodeContextBuilder:
    """
    Builds context observations for strategy network.
    
    Context includes:
    - Instance characteristics (n_jobs, n_machines, n_transbots)
    - Historical performance statistics
    - Recent preference effectiveness
    """
    
    def __init__(self):
        self.history = deque(maxlen=100)
        
    def build_context(self, env_config=None, recent_results=None):
        """
        Build context observation for strategy decision
        
        Returns:
            context_obs: 19-dim vector for strategy network input
        """
        context = np.zeros(19, dtype=np.float32)
        
        # Instance size features (normalized)
        if env_config:
            context[0] = env_config.get('n_jobs', 10) / 20.0  # Normalize by max expected
            context[1] = env_config.get('n_machines', 5) / 10.0
            context[2] = env_config.get('n_transbots', 3) / 6.0
        else:
            # Default values
            context[0] = 0.5
            context[1] = 0.5
            context[2] = 0.5
        
        # Historical performance features
        if recent_results and len(recent_results) > 0:
            recent_rewards = [r['reward'] for r in recent_results[-10:]]
            context[3] = np.mean(recent_rewards)  # Average recent reward
            context[4] = np.std(recent_rewards)   # Reward variance
            context[5] = np.max(recent_rewards)   # Best recent reward
            context[6] = np.min(recent_rewards)   # Worst recent reward
            
            # Preference effectiveness (if available)
            if 'preference' in recent_results[-1]:
                last_pref = recent_results[-1]['preference']
                context[7:10] = last_pref  # Last used preference
        
        # Time/progress features
        context[10] = len(self.history) / 100.0  # Training progress
        
        # Random features for exploration (will be learned to ignore if not useful)
        context[11:19] = np.random.randn(8) * 0.1
        
        return context
    
    def update_history(self, result):
        """Update history with episode result"""
        self.history.append(result)
