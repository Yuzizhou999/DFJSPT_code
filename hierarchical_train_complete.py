"""
Complete Hierarchical Training with Strategy Controller (Method 3)

Phase A Implementation - Full Training Version
- Strategy layer as meta-controller
- Tactical agents trained with RLlib PPO
- Checkpoint saving and recovery
- Training monitoring
"""

import sys
sys.path.append('.')

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
import numpy as np
import os
from datetime import datetime
import json
import torch  # For saving centralized critic

from DFJSPT import dfjspt_params
from DFJSPT.dfjspt_env import DfjsptMaEnv
from DFJSPT.strategy_controller import StrategyController, EpisodeContextBuilder

# Import centralized critic integration if enabled
if dfjspt_params.use_centralized_critic:
    from DFJSPT.centralized_critic_integration import (
        create_and_set_centralized_critic,
        register_all_models_with_catalog,
        get_custom_model_name_for_agent
    )
    print("✅ Centralized Critic enabled")
else:
    from DFJSPT.dfjspt_agent_model import (
        JobActionMaskModel, 
        MachineActionMaskModel, 
        TransbotActionMaskModel
    )
    print("ℹ️  Using decentralized critics")

print("="*80)
print("Hierarchical DFJSP-T Training (Method 3 - Strategy Meta-Controller)")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Problem size: {dfjspt_params.n_jobs}J * {dfjspt_params.n_machines}M * {dfjspt_params.n_transbots}T")
print(f"Multi-objective: {dfjspt_params.use_multi_objective_reward}")
print(f"GNN Encoder: {dfjspt_params.use_gnn_encoder}")
print(f"Centralized Critic: {dfjspt_params.use_centralized_critic}")
print("="*80)

# Configuration
NUM_ITERATIONS = dfjspt_params.stop_iters  # 优化1: 使用配置文件中的轮次设置
CHECKPOINT_INTERVAL = 5
STRATEGY_UPDATE_FREQUENCY = 5  # episodes per preference update
EXPLORATION_EPSILON_START = 0.3
EXPLORATION_EPSILON_END = 0.05
EXPLORATION_DECAY = 0.95

# Best model tracking
best_reward = float('-inf')
best_iteration = 0
best_checkpoint_dir = None

# Create centralized critic if enabled (must be done BEFORE model registration)
centralized_critic = None
if dfjspt_params.use_centralized_critic:
    print("\n📊 Creating centralized critic...")
    centralized_critic = create_and_set_centralized_critic()
    if centralized_critic is not None:
        print(f"✅ Centralized critic created successfully")
        n_params = sum(p.numel() for p in centralized_critic.parameters())
        print(f"   Parameters: {n_params:,}")
    else:
        print("⚠️  Centralized critic creation failed, falling back to decentralized")

# Register custom models
if dfjspt_params.use_centralized_critic and centralized_critic is not None:
    register_all_models_with_catalog(ModelCatalog)
else:
    ModelCatalog.register_custom_model("job_agent_model", JobActionMaskModel)
    ModelCatalog.register_custom_model("machine_agent_model", MachineActionMaskModel)
    ModelCatalog.register_custom_model("transbot_agent_model", TransbotActionMaskModel)
    print("📋 Custom models registered (decentralized)")

# Initialize Ray
ray.init(ignore_reinit_error=True, num_cpus=4)
print(" Ray initialized")

# Create example env to get spaces
example_env = DfjsptMaEnv({"train_or_eval_or_test": "train"})

# Create result directory
result_dir = os.path.join(
    "DFJSPT/training_results",
    f"hierarchical_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)
os.makedirs(result_dir, exist_ok=True)
print(f" Results will be saved to: {result_dir}")

# Create tactical PPO config (3 agents, no strategy in RLlib)
config = (
    PPOConfig()
    .environment(
        env=DfjsptMaEnv,
        env_config={"train_or_eval_or_test": "train"},
    )
    .framework("torch")
    .resources(
        num_gpus=1,
        num_cpus_for_local_worker=2,
    )
    .rollouts(
        num_rollout_workers=dfjspt_params.num_workers,
        num_envs_per_worker=dfjspt_params.num_envs_per_worker,
    )
    .training(
        train_batch_size=4000,  # 优化: 增加batch size以更好利用GPU
        sgd_minibatch_size=256,
        num_sgd_iter=10,
        lr=5e-4,  # 初始学习率
        entropy_coeff=0.01,  # 优化: 初期增加探索
        # 优化: 学习率调度 (基于timesteps)
        lr_schedule=[
            [0, 5e-4],       # 0轮: 初始学习率
            [80000, 3e-4],   # 推迟退火
            [160000, 2e-4],  # 保守降低，不要到1e-4
        ],
        # 优化: 熵系数调度 (减少后期探索)
        entropy_coeff_schedule=[
            [0, 0.01],
            [80000, 0.005],
            [160000, 0.003],  # 不要到0.001
        ],
        clip_param=0.2,  # 优化: 更保守的PPO裁剪
        grad_clip=0.5,  # 优化: 梯度裁剪防止爆炸
        use_gae=True,  # 优化: 启用GAE
        lambda_=0.95,  # 优化: GAE参数
    )
    .multi_agent(
        policies={
            "policy_agent0": (
                None,
                example_env.observation_space["agent0"],
                example_env.action_space["agent0"],
                {"model": {
                    "custom_model": get_custom_model_name_for_agent("agent0") if dfjspt_params.use_centralized_critic else "job_agent_model",
                    "fcnet_hiddens": [256, 256],
                    "fcnet_activation": "tanh",
                }}),
            "policy_agent1": (
                None,
                example_env.observation_space["agent1"],
                example_env.action_space["agent1"],
                {"model": {
                    "custom_model": get_custom_model_name_for_agent("agent1") if dfjspt_params.use_centralized_critic else "machine_agent_model",
                    "fcnet_hiddens": [256, 256],
                    "fcnet_activation": "tanh",
                }}),
            "policy_agent2": (
                None,
                example_env.observation_space["agent2"],
                example_env.action_space["agent2"],
                {"model": {
                    "custom_model": get_custom_model_name_for_agent("agent2") if dfjspt_params.use_centralized_critic else "transbot_agent_model",
                    "fcnet_hiddens": [128, 128],
                    "fcnet_activation": "tanh",
                }}),
        },
        policy_mapping_fn=lambda agent_id, episode=None, worker=None, **kwargs: f"policy_{agent_id}",
    )
    .debugging(log_level="WARN")
)

print("\nBuilding RLlib algorithm...")
algorithm = config.build()
print(" Algorithm built successfully")

# Create strategy controller
print("\nCreating strategy controller...")
strategy = StrategyController(
    obs_dim=19,
    action_dim=3,
    hidden_dim=128,
    learning_rate=3e-4,
    gamma=0.99,
    buffer_size=1000,
)
context_builder = EpisodeContextBuilder()
print(" Strategy controller created")

# Training tracking
training_history = []
best_reward = float('-inf')
current_preference = np.array([1/3, 1/3, 1/3])
episodes_with_current_preference = 0
exploration_epsilon = EXPLORATION_EPSILON_START

print("\n" + "="*80)
print(f"Starting training for {NUM_ITERATIONS} iterations")
print(f"Strategy update frequency: every {STRATEGY_UPDATE_FREQUENCY} episodes")
print(f"Checkpoint interval: every {CHECKPOINT_INTERVAL} iterations")
print("="*80 + "\n")

# Training loop
for iteration in range(NUM_ITERATIONS):
    iteration_start_time = datetime.now()
    
    print(f"\n{'='*80}")
    print(f"Iteration {iteration + 1}/{NUM_ITERATIONS}")
    print(f"{'='*80}")
    
    # Update exploration epsilon
    exploration_epsilon = max(
        EXPLORATION_EPSILON_END,
        EXPLORATION_EPSILON_START * (EXPLORATION_DECAY ** iteration)
    )
    
    # Check if we should update preference
    if episodes_with_current_preference >= STRATEGY_UPDATE_FREQUENCY or iteration == 0:
        # Build context for strategy decision
        context_obs = context_builder.build_context(
            env_config={"train_or_eval_or_test": "train"},
            recent_results=training_history[-10:] if training_history else None
        )
        
        # Get new preference from strategy controller
        new_preference = strategy.select_preference(
            context_obs,
            exploration=True,
            epsilon=exploration_epsilon
        )
        
        current_preference = new_preference
        episodes_with_current_preference = 0
        
        print(f"\nStrategy Update:")
        print(f"  Preference vector: {current_preference}")
        print(f"  Efficiency: {current_preference[0]:.3f}")
        print(f"  Cost:       {current_preference[1]:.3f}")
        print(f"  Delivery:   {current_preference[2]:.3f}")
        print(f"  Exploration ε: {exploration_epsilon:.3f}")
    
    # Note: In a more advanced version, we would set preference in worker envs
    # For now, envs use their default preference (can be enhanced later)
    
    # Train tactical agents
    print(f"\nTraining tactical agents...")
    result = algorithm.train()
    
    # Extract metrics
    episode_reward = result.get('episode_reward_mean', 0)
    episodes = result.get('episodes_this_iter', 0)
    timesteps = result.get('timesteps_total', 0)
    
    # Update counter
    episodes_with_current_preference += episodes
    
    # Print results
    print(f"\nOverall Results:")
    print(f"  Episode reward mean: {episode_reward:.4f}")
    print(f"  Episodes this iter:  {episodes}")
    print(f"  Timesteps total:     {timesteps}")
    
    # Extract custom metrics if available
    if 'sampler_results' in result and 'custom_metrics' in result['sampler_results']:
        metrics = result['sampler_results']['custom_metrics']
        makespan_mean = metrics.get('total_makespan_mean', None)
        drl_vs_rule = metrics.get('drl_minus_rule_mean', None)
        
        if makespan_mean is not None:
            print(f"  Makespan mean:       {makespan_mean:.2f}")
        if drl_vs_rule is not None:
            print(f"  DRL vs Rule:         {drl_vs_rule:+.2f}")
    
    # Display individual agent performance
    print(f"\nTactical Agents Performance:")
    
    # Agent 0 (Job selection)
    if 'policy_agent0' in result.get('info', {}).get('learner', {}).get('policy_agent0', {}):
        agent0_info = result['info']['learner']['policy_agent0']['learner_stats']
        print(f"  Agent0 (Job):")
        print(f"    Policy loss:     {agent0_info.get('policy_loss', 0):.6f}")
        print(f"    VF loss:         {agent0_info.get('vf_loss', 0):.6f}")
        print(f"    Entropy:         {agent0_info.get('entropy', 0):.6f}")
    
    # Agent 1 (Machine selection)
    if 'policy_agent1' in result.get('info', {}).get('learner', {}).get('policy_agent1', {}):
        agent1_info = result['info']['learner']['policy_agent1']['learner_stats']
        print(f"  Agent1 (Machine):")
        print(f"    Policy loss:     {agent1_info.get('policy_loss', 0):.6f}")
        print(f"    VF loss:         {agent1_info.get('vf_loss', 0):.6f}")
        print(f"    Entropy:         {agent1_info.get('entropy', 0):.6f}")
    
    # Agent 2 (Transbot selection)
    if 'policy_agent2' in result.get('info', {}).get('learner', {}).get('policy_agent2', {}):
        agent2_info = result['info']['learner']['policy_agent2']['learner_stats']
        print(f"  Agent2 (Transbot):")
        print(f"    Policy loss:     {agent2_info.get('policy_loss', 0):.6f}")
        print(f"    VF loss:         {agent2_info.get('vf_loss', 0):.6f}")
        print(f"    Entropy:         {agent2_info.get('entropy', 0):.6f}")
    
    # Learning rates
    if 'info' in result and 'learner' in result['info']:
        for policy_id in ['policy_agent0', 'policy_agent1', 'policy_agent2']:
            if policy_id in result['info']['learner']:
                lr = result['info']['learner'][policy_id]['learner_stats'].get('cur_lr', None)
                if lr is not None:
                    agent_name = policy_id.replace('policy_agent', 'Agent')
                    print(f"  {agent_name} learning rate: {lr:.2e}")
    
    # Store experience for strategy learning
    iteration_result = {
        'iteration': iteration + 1,
        'preference': current_preference.tolist(),
        'reward': episode_reward,
        'episodes': episodes,
        'timesteps': timesteps,
        'exploration_epsilon': exploration_epsilon,
    }
    
    # Add agent-specific metrics if available
    if 'info' in result and 'learner' in result['info']:
        agent_metrics = {}
        for policy_id in ['policy_agent0', 'policy_agent1', 'policy_agent2']:
            if policy_id in result['info']['learner']:
                stats = result['info']['learner'][policy_id]['learner_stats']
                agent_name = policy_id.replace('policy_', '')
                agent_metrics[agent_name] = {
                    'policy_loss': float(stats.get('policy_loss', 0)),
                    'vf_loss': float(stats.get('vf_loss', 0)),
                    'entropy': float(stats.get('entropy', 0)),
                    'cur_lr': float(stats.get('cur_lr', 0)),
                }
        if agent_metrics:
            iteration_result['agent_metrics'] = agent_metrics
    
    # Add custom metrics if available
    if 'sampler_results' in result and 'custom_metrics' in result['sampler_results']:
        custom = result['sampler_results']['custom_metrics']
        iteration_result['makespan_mean'] = custom.get('total_makespan_mean', None)
        iteration_result['drl_vs_rule'] = custom.get('drl_minus_rule_mean', None)
    
    training_history.append(iteration_result)
    
    # 优化: 跟踪最佳模型
    if episode_reward > best_reward:
        best_reward = episode_reward
        best_iteration = iteration + 1
        print(f"\n🏆 New best reward: {best_reward:.4f} (iteration {best_iteration})")
        
        # 立即保存最佳模型
        best_checkpoint_dir = os.path.join(result_dir, "best_checkpoint")
        os.makedirs(best_checkpoint_dir, exist_ok=True)
        
        algorithm.save(best_checkpoint_dir)
        strategy_path = os.path.join(best_checkpoint_dir, "strategy_controller.pt")
        strategy.save(strategy_path)
        
        if dfjspt_params.use_centralized_critic and centralized_critic is not None:
            critic_path = os.path.join(best_checkpoint_dir, "centralized_critic.pt")
            torch.save(centralized_critic.state_dict(), critic_path)
        
        # 保存最佳模型的元信息
        best_info_path = os.path.join(best_checkpoint_dir, "best_model_info.json")
        with open(best_info_path, 'w') as f:
            json.dump({
                'iteration': best_iteration,
                'reward': float(best_reward),
                'preference': current_preference.tolist(),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2)
        
        print(f"  Best model saved to: {best_checkpoint_dir}")
    
    # Build next context
    next_context = context_builder.build_context(
        env_config={"train_or_eval_or_test": "train"},
        recent_results=training_history[-10:]
    )
    
    # Store experience in strategy buffer
    strategy.store_experience(
        context_obs=context_obs if 'context_obs' in locals() else np.zeros(19),
        preference=current_preference,
        reward=episode_reward,
        next_context_obs=next_context,
        done=False
    )
    
    # Update strategy controller periodically
    if len(strategy.buffer) >= 32 and (iteration + 1) % 5 == 0:
        loss = strategy.update(batch_size=32)
        if loss is not None:
            print(f"\nStrategy Learning:")
            print(f"  Buffer size: {len(strategy.buffer)}")
            print(f"  Loss: {loss:.6f}")
    
    # Save checkpoint
    if (iteration + 1) % CHECKPOINT_INTERVAL == 0:
        checkpoint_dir = os.path.join(result_dir, f"checkpoint_{iteration+1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save tactical policies
        tactical_checkpoint = algorithm.save(checkpoint_dir)
        
        # Save strategy controller
        strategy_path = os.path.join(checkpoint_dir, "strategy_controller.pt")
        strategy.save(strategy_path)
        
        # Save centralized critic if enabled
        if dfjspt_params.use_centralized_critic and centralized_critic is not None:
            critic_path = os.path.join(checkpoint_dir, "centralized_critic.pt")
            torch.save(centralized_critic.state_dict(), critic_path)
        
        # Save training history
        history_path = os.path.join(checkpoint_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        print(f"\nCheckpoint saved:")
        print(f"  Location: {checkpoint_dir}")
    
    # Calculate iteration time
    iteration_time = (datetime.now() - iteration_start_time).total_seconds()
    print(f"\n⏱Iteration time: {iteration_time:.1f}s")

# Final checkpoint
print(f"\n{'='*80}")
print("Training Complete!")
print(f"{'='*80}")

final_checkpoint_dir = os.path.join(result_dir, "final_checkpoint")
os.makedirs(final_checkpoint_dir, exist_ok=True)

tactical_checkpoint = algorithm.save(final_checkpoint_dir)
strategy_path = os.path.join(final_checkpoint_dir, "strategy_controller.pt")
strategy.save(strategy_path)

# Save centralized critic if enabled
if dfjspt_params.use_centralized_critic and centralized_critic is not None:
    critic_path = os.path.join(final_checkpoint_dir, "centralized_critic.pt")
    torch.save(centralized_critic.state_dict(), critic_path)
    print(f"✅ Centralized critic saved to: {critic_path}")

history_path = os.path.join(result_dir, "complete_training_history.json")
with open(history_path, 'w') as f:
    json.dump(training_history, f, indent=2)

print(f"\nFinal checkpoint saved to: {final_checkpoint_dir}")
print(f"Training history saved to: {history_path}")

# Print summary
print(f"\n{'='*80}")
print("Training Summary")
print(f"{'='*80}")
print(f"Total iterations:     {NUM_ITERATIONS}")
print(f"Total timesteps:      {timesteps}")
print(f"Total episodes:       {sum(h['episodes'] for h in training_history)}")
print(f"\nReward Statistics:")
rewards = [h['reward'] for h in training_history]
print(f"  Initial reward:     {rewards[0]:.4f}")
print(f"  Final reward:       {rewards[-1]:.4f}")
print(f"  Best reward:        {max(rewards):.4f} (iteration {rewards.index(max(rewards))+1})")
print(f"  🏆 Best model saved at iteration {best_iteration} with reward {best_reward:.4f}")
print(f"  Average (last 10):  {np.mean(rewards[-10:]):.4f}")
print(f"  Average (last 20):  {np.mean(rewards[-20:]):.4f}")
print(f"  Improvement:        {rewards[-1] - rewards[0]:.4f} ({(rewards[-1]/rewards[0]-1)*100:.1f}%)")

print(f"\nStrategy Controller:")
print(f"  Buffer size:        {len(strategy.buffer)}")
print(f"  Final preference:   {current_preference}")
print(f"    Efficiency:       {current_preference[0]:.3f}")
print(f"    Cost:             {current_preference[1]:.3f}")
print(f"    Delivery:         {current_preference[2]:.3f}")

# Calculate preference statistics
preferences = np.array([h['preference'] for h in training_history])
print(f"\nPreference Evolution (last 10 iterations):")
print(f"  Efficiency:         {preferences[-10:,0].mean():.3f} ± {preferences[-10:,0].std():.3f}")
print(f"  Cost:               {preferences[-10:,1].mean():.3f} ± {preferences[-10:,1].std():.3f}")
print(f"  Delivery:           {preferences[-10:,2].mean():.3f} ± {preferences[-10:,2].std():.3f}")

# Agent performance summary
if 'agent_metrics' in training_history[-1]:
    print(f"\nTactical Agents Final Performance:")
    final_metrics = training_history[-1]['agent_metrics']
    for agent_name, metrics in final_metrics.items():
        print(f"  {agent_name}:")
        print(f"    Policy loss:      {metrics['policy_loss']:.6f}")
        print(f"    VF loss:          {metrics['vf_loss']:.6f}")
        print(f"    Entropy:          {metrics['entropy']:.6f}")

# Custom metrics summary
if 'makespan_mean' in training_history[-1] and training_history[-1]['makespan_mean'] is not None:
    makespans = [h.get('makespan_mean', 0) for h in training_history if 'makespan_mean' in h]
    print(f"\nMakespan Statistics:")
    print(f"  Final:              {makespans[-1]:.2f}")
    print(f"  Average (last 10):  {np.mean(makespans[-10:]):.2f}")
    if 'drl_vs_rule' in training_history[-1]:
        drl_vs_rules = [h.get('drl_vs_rule', 0) for h in training_history if 'drl_vs_rule' in h]
        print(f"  DRL vs Rule (avg):  {np.mean(drl_vs_rules):.2f}")

print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*80}")

ray.shutdown()
print("\nTraining completed successfully!")
