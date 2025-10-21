
# training params
as_test = False
framework = "torch"
local_mode = False
use_tune = True  
use_custom_loss = True
il_loss_weight = 10.0
stop_iters = 100
stop_timesteps = 100000000000
stop_reward = 5
num_workers = 4
num_gpu = 1
num_envs_per_worker = 1


max_n_jobs = 20
n_jobs_is_fixed = True
n_jobs = 20
n_operations_is_n_machines = False
min_n_operations = 5
max_n_operations = 5
consider_job_insert = False
new_arrival_jobs = 5
earliest_arrive_time = 30
latest_arrive_time = 300

max_n_machines = 5
min_prcs_time = 1
max_prcs_time = 100
n_machines_is_fixed = True
n_machines = 5
is_fully_flexible = False
min_compatible_machines = 1
time_for_compatible_machines_are_same = False
time_viration_range = 5

max_n_transbots = 3
min_tspt_time = 1
max_tspt_time = 10
loaded_transport_time_scale = 1.5
n_transbots_is_fixed = True
n_transbots = 3

all_machines_are_perfect = False
min_quality = 0.1
# normalized_scale = max_n_operations * max_prcs_time

n_instances = 5200
n_instances_for_training = 5000
n_instances_for_evaluation = 100
n_instances_for_testing = 100
instance_generator_seed = 1000
layout_seed = 0

# env params
perform_left_shift_if_possible = True

# instance selection params
randomly_select_instance = True
current_instance_id = 0
imitation_env_count = 0
env_count = 0


# render params
JobAsAction = True
gantt_y_axis = "nJob"
drawMachineToPrcsEdges = True
default_visualisations = None


# ===== Hierarchical MARL params =====
# Strategy layer configuration
use_hierarchical_framework = True  # RE-ENABLED after fixing strategy observation frequency
strategy_update_frequency = 1  # How often strategy layer makes decisions (in episodes)
strategy_action_continuous = True  # True: continuous preference vector, False: discrete choices
strategy_action_dim = 3  # Dimension of preference vector [efficiency, cost, delivery]

# Multi-objective reward configuration
use_multi_objective_reward = True  # RE-ENABLED after fixing strategy observation frequency
reward_normalization_method = "ema"  # Options: "ema", "minmax", "none"
ema_alpha = 0.1  # EMA smoothing factor for reward normalization

# Reward component weights (for computing sub-rewards)
efficiency_weight = 1.0  # Weight for efficiency reward (makespan, utilization)
cost_weight = 1.0  # Weight for cost reward (energy, transport, wear)
delivery_weight = 1.0  # Weight for delivery reward (tardiness penalty)

# Cost calculation parameters
energy_cost_per_unit_time = 0.1  # Cost per unit time machine running
transport_cost_per_unit_time = 0.05  # Cost per unit time transbot moving
machine_wear_cost = 0.01  # Cost per operation for machine wear

# Delivery parameters
delivery_due_time_factor = 1.5  # Due time = mean_processing_time * this factor
tardiness_penalty_factor = 2.0  # Penalty multiplier for late jobs

# Potential-based reward shaping
use_potential_shaping = False  # Enable potential-based reward shaping
potential_gamma = 0.99  # Discount factor for potential shaping

# GNN encoder configuration
use_gnn_encoder = True  # Enable GNN encoder (requires PyTorch Geometric)
gnn_hidden_dim = 128  # Hidden dimension for GNN
gnn_num_layers = 3  # Number of GNN layers (优化4: 保持3层捕获更深依赖)
gnn_type = "gat"  # GNN type: "gcn", "gat", "sage"
gnn_pooling = "attention"  # Pooling method: "mean", "max", "attention"
gnn_num_heads = 4  # Number of attention heads for GAT
gnn_dropout = 0.1  # Dropout rate for GNN layers

# Multi-objective reward normalization
use_reward_normalization = True  # Enable dynamic reward normalization
normalization_method = "ema"  # Normalization method: "standardize", "minmax", "ema"
normalization_momentum = 0.99  # EMA momentum for normalization (higher = slower adaptation)
normalization_epsilon = 1e-8  # Small constant for numerical stability

# Centralized Critic (CTDE)
use_centralized_critic = True  # Enable centralized critic for value estimation
centralized_critic_hidden_dim = 256  # 优化2: 增加网络容量 (原: 128)
centralized_critic_use_gnn = True  # Use GNN encoder in centralized critic (if use_gnn_encoder=True)
centralized_critic_lr = 2e-4  # 优化3: 独立Critic学习率 (比policy lr更保守)




