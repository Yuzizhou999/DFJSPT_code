"""
Dynamic Reward Normalization for Multi-Objective DFJSP-T

Implements running statistics tracking and adaptive normalization for three reward components:
1. Efficiency (makespan-related)
2. Cost (machine quality, transbot quality)
3. Delivery (due date, tardiness)

This ensures no single objective dominates the learning process during early training.

Reference:
- Running mean and variance normalization
- Per-component adaptive scaling
- Exponential moving average for stability
"""

import numpy as np
from typing import Dict, Tuple, Optional
import pickle
import os


class RunningStatistics:
    """
    Tracks running statistics (mean, variance, min, max) for a single reward component.
    
    Uses Welford's online algorithm for numerical stability.
    """
    
    def __init__(self, epsilon=1e-8, momentum=0.99):
        """
        Args:
            epsilon: Small constant for numerical stability
            momentum: Exponential moving average momentum (0.99 means slow adaptation)
        """
        self.epsilon = epsilon
        self.momentum = momentum
        
        # Running statistics
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0  # Sum of squared differences from mean
        self.min_value = float('inf')
        self.max_value = float('-inf')
        
        # Exponential moving average
        self.ema_mean = 0.0
        self.ema_var = 1.0
    
    def update(self, value: float):
        """
        Update statistics with new value using Welford's algorithm.
        
        Args:
            value: New observation
        """
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.M2 += delta * delta2
        
        # Update min/max
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)
        
        # Update EMA
        if self.count == 1:
            self.ema_mean = value
            self.ema_var = 0.0
        else:
            self.ema_mean = self.momentum * self.ema_mean + (1 - self.momentum) * value
            self.ema_var = self.momentum * self.ema_var + (1 - self.momentum) * (value - self.ema_mean) ** 2
    
    def get_variance(self) -> float:
        """Get sample variance."""
        if self.count < 2:
            return 0.0
        return self.M2 / (self.count - 1)
    
    def get_std(self) -> float:
        """Get standard deviation."""
        return np.sqrt(self.get_variance() + self.epsilon)
    
    def normalize(self, value: float, method: str = "standardize") -> float:
        """
        Normalize value using current statistics.
        
        Args:
            value: Value to normalize
            method: "standardize" (z-score), "minmax" (0-1 scaling), or "ema" (using EMA stats)
        
        Returns:
            Normalized value
        """
        if method == "standardize":
            if self.count < 2:
                return value
            return (value - self.mean) / (self.get_std() + self.epsilon)
        
        elif method == "minmax":
            if self.count < 2 or self.max_value == self.min_value:
                return value
            return (value - self.min_value) / (self.max_value - self.min_value + self.epsilon)
        
        elif method == "ema":
            if self.count < 2:
                return value
            std = np.sqrt(self.ema_var + self.epsilon)
            return (value - self.ema_mean) / (std + self.epsilon)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def denormalize(self, normalized_value: float, method: str = "standardize") -> float:
        """
        Denormalize value back to original scale.
        
        Args:
            normalized_value: Normalized value
            method: Same method used for normalization
        
        Returns:
            Denormalized value
        """
        if method == "standardize":
            return normalized_value * self.get_std() + self.mean
        elif method == "minmax":
            return normalized_value * (self.max_value - self.min_value) + self.min_value
        elif method == "ema":
            std = np.sqrt(self.ema_var + self.epsilon)
            return normalized_value * std + self.ema_mean
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def get_stats(self) -> Dict:
        """Get current statistics as dictionary."""
        return {
            'count': self.count,
            'mean': self.mean,
            'std': self.get_std(),
            'variance': self.get_variance(),
            'min': self.min_value,
            'max': self.max_value,
            'ema_mean': self.ema_mean,
            'ema_std': np.sqrt(self.ema_var + self.epsilon),
        }
    
    def reset(self):
        """Reset all statistics."""
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min_value = float('inf')
        self.max_value = float('-inf')
        self.ema_mean = 0.0
        self.ema_var = 1.0


class MultiObjectiveRewardNormalizer:
    """
    Normalizes multiple reward components to prevent any single objective from dominating.
    
    Tracks separate statistics for:
    - Efficiency reward (makespan, throughput)
    - Cost reward (quality costs)
    - Delivery reward (tardiness, due date violations)
    """
    
    def __init__(
        self,
        component_names: Tuple[str, ...] = ('efficiency', 'cost', 'delivery'),
        normalization_method: str = "ema",  # "standardize", "minmax", or "ema"
        epsilon: float = 1e-8,
        momentum: float = 0.99,
        enable_normalization: bool = True,
    ):
        """
        Args:
            component_names: Names of reward components
            normalization_method: Method for normalization
            epsilon: Small constant for numerical stability
            momentum: EMA momentum
            enable_normalization: Whether to enable normalization (can disable for testing)
        """
        self.component_names = component_names
        self.normalization_method = normalization_method
        self.epsilon = epsilon
        self.momentum = momentum
        self.enable_normalization = enable_normalization
        
        # Create statistics tracker for each component
        self.statistics = {
            name: RunningStatistics(epsilon=epsilon, momentum=momentum)
            for name in component_names
        }
        
        # Track history for analysis
        self.history = {name: [] for name in component_names}
        self.normalized_history = {name: [] for name in component_names}
    
    def update_and_normalize(
        self,
        rewards: Dict[str, float],
        update_stats: bool = True,
    ) -> Dict[str, float]:
        """
        Update statistics and normalize rewards.
        
        Args:
            rewards: Dict of {component_name: reward_value}
            update_stats: Whether to update running statistics (False during evaluation)
        
        Returns:
            Normalized rewards dict
        """
        normalized_rewards = {}
        
        for name in self.component_names:
            if name not in rewards:
                raise ValueError(f"Missing reward component: {name}")
            
            value = rewards[name]
            
            # Update statistics
            if update_stats:
                self.statistics[name].update(value)
                self.history[name].append(value)
            
            # Normalize
            if self.enable_normalization:
                normalized_value = self.statistics[name].normalize(value, self.normalization_method)
            else:
                normalized_value = value
            
            normalized_rewards[name] = normalized_value
            
            if update_stats:
                self.normalized_history[name].append(normalized_value)
        
        return normalized_rewards
    
    def get_statistics_summary(self) -> Dict:
        """
        Get summary of all component statistics.
        
        Returns:
            Dict with statistics for each component
        """
        summary = {}
        for name, stats in self.statistics.items():
            summary[name] = stats.get_stats()
        return summary
    
    def print_statistics(self):
        """Print statistics for all components."""
        print("\n" + "="*80)
        print("Multi-Objective Reward Statistics")
        print("="*80)
        
        for name, stats in self.statistics.items():
            stat_dict = stats.get_stats()
            print(f"\n{name.upper()}:")
            print(f"  Count:     {stat_dict['count']}")
            print(f"  Mean:      {stat_dict['mean']:.4f}")
            print(f"  Std:       {stat_dict['std']:.4f}")
            print(f"  Min:       {stat_dict['min']:.4f}")
            print(f"  Max:       {stat_dict['max']:.4f}")
            print(f"  EMA Mean:  {stat_dict['ema_mean']:.4f}")
            print(f"  EMA Std:   {stat_dict['ema_std']:.4f}")
        
        print("="*80)
    
    def check_balance(self, threshold: float = 2.0) -> Dict[str, bool]:
        """
        Check if reward components are balanced (no single component dominates).
        
        Args:
            threshold: Maximum allowed ratio between component standard deviations
        
        Returns:
            Dict of {component_name: is_balanced}
        """
        stds = {name: stats.get_std() for name, stats in self.statistics.items()}
        max_std = max(stds.values())
        min_std = min(stds.values()) + self.epsilon
        
        ratio = max_std / min_std
        is_balanced = ratio <= threshold
        
        balance_status = {}
        for name in self.component_names:
            balance_status[name] = stds[name] <= (threshold * min_std)
        
        balance_status['overall_balanced'] = is_balanced
        balance_status['std_ratio'] = ratio
        
        return balance_status
    
    def save(self, filepath: str):
        """Save normalizer state to file."""
        state = {
            'component_names': self.component_names,
            'normalization_method': self.normalization_method,
            'epsilon': self.epsilon,
            'momentum': self.momentum,
            'enable_normalization': self.enable_normalization,
            'statistics': {
                name: {
                    'count': stats.count,
                    'mean': stats.mean,
                    'M2': stats.M2,
                    'min_value': stats.min_value,
                    'max_value': stats.max_value,
                    'ema_mean': stats.ema_mean,
                    'ema_var': stats.ema_var,
                }
                for name, stats in self.statistics.items()
            },
            'history': self.history,
            'normalized_history': self.normalized_history,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"Normalizer saved to {filepath}")
    
    def load(self, filepath: str):
        """Load normalizer state from file."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.component_names = state['component_names']
        self.normalization_method = state['normalization_method']
        self.epsilon = state['epsilon']
        self.momentum = state['momentum']
        self.enable_normalization = state['enable_normalization']
        
        # Restore statistics
        for name, stat_dict in state['statistics'].items():
            if name not in self.statistics:
                self.statistics[name] = RunningStatistics(self.epsilon, self.momentum)
            
            stats = self.statistics[name]
            stats.count = stat_dict['count']
            stats.mean = stat_dict['mean']
            stats.M2 = stat_dict['M2']
            stats.min_value = stat_dict['min_value']
            stats.max_value = stat_dict['max_value']
            stats.ema_mean = stat_dict['ema_mean']
            stats.ema_var = stat_dict['ema_var']
        
        self.history = state.get('history', {name: [] for name in self.component_names})
        self.normalized_history = state.get('normalized_history', {name: [] for name in self.component_names})
        
        print(f"Normalizer loaded from {filepath}")


# Testing and example usage
if __name__ == "__main__":
    print("Testing Multi-Objective Reward Normalizer...")
    
    # Create normalizer
    normalizer = MultiObjectiveRewardNormalizer(
        component_names=('efficiency', 'cost', 'delivery'),
        normalization_method="ema",
        momentum=0.99,
    )
    
    # Simulate rewards with different scales and distributions
    np.random.seed(42)
    n_steps = 1000
    
    for step in range(n_steps):
        # Simulate rewards with very different scales
        rewards = {
            'efficiency': -np.random.exponential(100),  # Large negative values (makespan)
            'cost': -np.random.uniform(0.1, 1.0),  # Small negative values (quality cost)
            'delivery': -np.random.gamma(2, 5),  # Medium negative values (tardiness)
        }
        
        normalized = normalizer.update_and_normalize(rewards)
        
        if step % 200 == 0:
            print(f"\nStep {step}:")
            print(f"  Raw: Eff={rewards['efficiency']:.2f}, Cost={rewards['cost']:.2f}, Deliv={rewards['delivery']:.2f}")
            print(f"  Norm: Eff={normalized['efficiency']:.2f}, Cost={normalized['cost']:.2f}, Deliv={normalized['delivery']:.2f}")
    
    # Print final statistics
    normalizer.print_statistics()
    
    # Check balance
    balance = normalizer.check_balance(threshold=2.0)
    print("\nBalance Check:")
    for key, value in balance.items():
        print(f"  {key}: {value}")
    
    # Test save/load
    normalizer.save("test_normalizer.pkl")
    
    new_normalizer = MultiObjectiveRewardNormalizer()
    new_normalizer.load("test_normalizer.pkl")
    
    print("\n✅ All tests passed!")
    
    # Clean up
    if os.path.exists("test_normalizer.pkl"):
        os.remove("test_normalizer.pkl")
