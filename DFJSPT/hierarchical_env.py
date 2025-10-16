"""
Hierarchical wrapper for DFJSPT environment to support strategy layer
"""
import numpy as np
from gymnasium import spaces
from DFJSPT.dfjspt_env import DfjsptMaEnv
from DFJSPT import dfjspt_params


class HierarchicalDfjsptEnv(DfjsptMaEnv):
    """
    Wrapper around DfjsptMaEnv that adds a strategy layer.
    
    The strategy layer outputs preference vectors that guide the tactical layer.
    Strategy layer makes decisions at the beginning of each episode.
    """
    
    def __init__(self, env_config):
        super().__init__(env_config)
        
        # Add strategy agent to the agents set
        self.agents = {"strategy", "agent0", "agent1", "agent2"}
        self._agent_ids = set(self.agents)
        
        # Strategy observation space
        strategy_obs_dim = 19
        strategy_obs_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(strategy_obs_dim,),
            dtype=np.float32
        )
        
        # Strategy action space (discrete for now)
        if dfjspt_params.strategy_action_continuous:
            strategy_action_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(dfjspt_params.strategy_action_dim,),
                dtype=np.float32
            )
        else:
            # 7 discrete preference combinations
            strategy_action_space = spaces.Discrete(7)
        
        # Add strategy to observation and action spaces
        self.observation_space["strategy"] = strategy_obs_space
        self.action_space["strategy"] = strategy_action_space
        
        # Track whether strategy needs to make a decision
        self.strategy_needs_decision = True
        self.steps_since_strategy_update = 0
        
    def reset(self, seed=None, options=None):
        """Reset environment and initialize - strategy acts ONLY at step 0"""
        obs, info = super().reset(seed=seed, options=None)
        
        # Initialize with default balanced preference
        self.current_preference_vector = np.array([1/3, 1/3, 1/3])
        self.episode_step = 0
        
        # CRITICAL FIX: Strategy ONLY acts at step 0
        # Parent reset() returns obs for only the first tactical agent (agent0 at stage 0)
        # We add strategy obs ONLY at reset so it makes ONE decision per episode
        # After step 0, strategy becomes passive observer without acting
        strategy_obs = self.get_strategy_obs()
        obs["strategy"] = strategy_obs
        
        return obs, info
    
    def step(self, action):
        """
        Step function that handles both strategy and tactical decisions.
        
        Strategy agent provides preference at every step, but we only update 
        the preference vector at specific times (e.g., episode start).
        """
        # Extract actions
        strategy_action = action.get("strategy")
        
        # Update preference vector if strategy action is provided
        # For now, update at first step of each episode
        if self.episode_step == 0 and strategy_action is not None:
            # Convert strategy action to preference vector
            if dfjspt_params.strategy_action_continuous:
                # Continuous action: apply softmax to get preference vector
                preference_logits = strategy_action
                exp_logits = np.exp(preference_logits - np.max(preference_logits))
                preference_vector = exp_logits / exp_logits.sum()
            else:
                # Discrete action: map to predefined preferences
                preference_map = {
                    0: np.array([1/3, 1/3, 1/3]),  # Balanced
                    1: np.array([0.8, 0.1, 0.1]),  # Efficiency-focused
                    2: np.array([0.1, 0.8, 0.1]),  # Cost-focused
                    3: np.array([0.1, 0.1, 0.8]),  # Delivery-focused
                    4: np.array([0.6, 0.3, 0.1]),  # Efficiency+Cost
                    5: np.array([0.6, 0.1, 0.3]),  # Efficiency+Delivery
                    6: np.array([0.1, 0.6, 0.3]),  # Cost+Delivery
                }
                preference_vector = preference_map.get(strategy_action, np.array([1/3, 1/3, 1/3]))
            
            # Update environment's preference vector
            self.current_preference_vector = preference_vector
        
    def step(self, action):
        """
        Step function that handles both strategy and tactical decisions.
        
        ARCHITECTURE DECISION: Strategy observes every step to satisfy RLlib's consistency requirements,
        but ONLY makes meaningful decisions at step 0.
        
        RLlib requires that agents with rewards/terminated must have observations every step.
        Strategy gets obs+reward+terminated every step, but ignores action after step 0.
        
        In DfjsptMaEnv, tactical agents act in sequence (stage 0: agent0, stage 1: agent1, stage 2: agent2).
        Strategy sets preference at step 0, then passively observes (actions after step 0 are ignored).
        """
        # IMPORTANT: Only extract strategy action at step 0
        # After that, strategy actions are provided by RLlib but we ignore them
        if self.episode_step == 0 and "strategy" in action:
            strategy_action = action["strategy"]
            
            # Convert strategy action to preference vector
            if dfjspt_params.strategy_action_continuous:
                # Continuous action: apply softmax to get preference vector
                preference_logits = strategy_action
                exp_logits = np.exp(preference_logits - np.max(preference_logits))
                preference_vector = exp_logits / exp_logits.sum()
            else:
                # Discrete action: map to predefined preferences
                preference_map = {
                    0: np.array([1/3, 1/3, 1/3]),  # Balanced
                    1: np.array([0.8, 0.1, 0.1]),  # Efficiency-focused
                    2: np.array([0.1, 0.8, 0.1]),  # Cost-focused
                    3: np.array([0.1, 0.1, 0.8]),  # Delivery-focused
                    4: np.array([0.6, 0.3, 0.1]),  # Efficiency+Cost
                    5: np.array([0.6, 0.1, 0.3]),  # Efficiency+Delivery
                    6: np.array([0.1, 0.6, 0.3]),  # Cost+Delivery
                }
                preference_vector = preference_map.get(strategy_action, np.array([1/3, 1/3, 1/3]))
            
            # Update environment's preference vector
            self.current_preference_vector = preference_vector
        
        # Extract only tactical actions for parent class
        # Parent class expects only the tactical agent(s) that should act this step
        # In DfjsptMaEnv, only ONE tactical agent acts per step (based on stage)
        tactical_action = {k: v for k, v in action.items() if k in {"agent0", "agent1", "agent2"}}
        
        # Check if we have any tactical actions
        if not tactical_action:
            # This can happen when episode has ended and RLlib sends one more step
            # In this case, previous step should have set terminated["__all__"]=True
            # We just return terminal state for all agents
            obs = {}
            reward = {}
            terminated = {}
            truncated = {}
            
            # Strategy observation - use actual global state
            obs["strategy"] = self.get_strategy_obs()
            reward["strategy"] = 0.0
            terminated["strategy"] = True
            truncated["strategy"] = False
            
            # Tactical agents - return dummy observations matching their observation space
            # agent0, agent1, agent2 need Dict observations with action_mask and observation
            # Use parent class's feature dimensions
            for agent_id in ["agent0", "agent1", "agent2"]:
                # Return dummy observation in correct format
                # Get the observation space dimensions from parent class
                if agent_id == "agent0":
                    # Jobs: use parent class's dimensions
                    n_items = self.n_jobs  # Use actual number from parent
                    n_features = self.n_job_features  # From parent class
                    obs[agent_id] = {
                        "action_mask": np.zeros(n_items, dtype=np.int64),
                        "observation": np.zeros((n_items, n_features), dtype=np.float32)
                    }
                elif agent_id == "agent1":
                    # Machines
                    n_items = self.n_machines  # Use actual number from parent
                    n_features = self.n_machine_features  # From parent class
                    obs[agent_id] = {
                        "action_mask": np.zeros(n_items, dtype=np.int64),
                        "observation": np.zeros((n_items, n_features), dtype=np.float32)
                    }
                else:  # agent2
                    # Transbots
                    n_items = self.n_transbots  # Use actual number from parent
                    n_features = self.n_transbot_features  # From parent class
                    obs[agent_id] = {
                        "action_mask": np.zeros(n_items, dtype=np.int64),
                        "observation": np.zeros((n_items, n_features), dtype=np.float32)
                    }
                reward[agent_id] = 0.0
                terminated[agent_id] = True
                truncated[agent_id] = False
            
            terminated["__all__"] = True
            truncated["__all__"] = False
            info = {}
            return obs, reward, terminated, truncated, info
        
        # Process tactical agent actions through parent class
        obs, reward, terminated, truncated, info = super().step(tactical_action)
        
        # Increment episode step counter
        self.episode_step += 1
        
        # CRITICAL APPROACH CHANGE: Strategy observes EVERY step but only acts at step 0
        # This satisfies RLlib's requirement that agents with rewards/terminated must have consistent obs
        # Strategy gets a dummy observation after step 0 (but we already updated preference at step 0)
        strategy_obs = self.get_strategy_obs()
        obs["strategy"] = strategy_obs
        
        # CRITICAL: Ensure ALL agents (including strategy) have rewards at every step
        # This is required by RLlib - all agents must have consistent trajectory data
        if isinstance(reward, dict):
            tactical_rewards = [r for k, r in reward.items() if k.startswith("agent")]
            strategy_reward = np.mean(tactical_rewards) if tactical_rewards else 0.0
            reward["strategy"] = strategy_reward
            
            # Ensure all tactical agents have a reward (even if 0.0)
            # This prevents RLlib from thinking we have multiple trajectories
            for agent_id in ["agent0", "agent1", "agent2"]:
                if agent_id not in reward:
                    reward[agent_id] = 0.0
        
        # CRITICAL: Ensure all agents (INCLUDING STRATEGY) have terminated/truncated status
        # Strategy must have consistent episode boundaries like tactical agents
        if isinstance(terminated, dict):
            term_all = terminated.get("__all__", False)
            terminated["strategy"] = term_all
            
            # Ensure ALL tactical agents have terminated status
            # Even if they didn't act this step, they must have consistent episode boundaries
            for agent_id in ["agent0", "agent1", "agent2"]:
                if agent_id not in terminated:
                    terminated[agent_id] = term_all
        
        if isinstance(truncated, dict):
            trunc_all = truncated.get("__all__", False)
            truncated["strategy"] = trunc_all
            
            # Ensure ALL tactical agents have truncated status
            for agent_id in ["agent0", "agent1", "agent2"]:
                if agent_id not in truncated:
                    truncated[agent_id] = trunc_all
        
        # Reset episode step counter if episode ends
        if terminated.get("__all__", False):
            self.episode_step = 0
        
        return obs, reward, terminated, truncated, info


