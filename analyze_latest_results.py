"""
详细分析最新的分层训练结果 (100轮训练)
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 读取最新训练结果
results_dir = Path("DFJSPT/training_results/hierarchical_v3_20251016_141634")
history_file = results_dir / "complete_training_history.json"

with open(history_file, 'r') as f:
    history = json.load(f)

print("=" * 80)
print("最新训练结果分析 (100轮训练)")
print("=" * 80)
print(f"\n训练目录: {results_dir.name}")
print(f"总迭代次数: {len(history)}")
print(f"总时间步数: {history[-1]['timesteps']}")

# 1. 奖励分析
rewards = [h['reward'] for h in history]
print("\n" + "=" * 80)
print("1. 奖励曲线分析")
print("=" * 80)
print(f"初始奖励: {rewards[0]:.4f}")
print(f"最终奖励: {rewards[-1]:.4f}")
print(f"最佳奖励: {max(rewards):.4f} (第 {np.argmax(rewards) + 1} 轮)")
print(f"最差奖励: {min(rewards):.4f} (第 {np.argmin(rewards) + 1} 轮)")
print(f"总体改进: {rewards[-1] - rewards[0]:.4f} ({(rewards[-1] - rewards[0]) / abs(rewards[0]) * 100:.2f}%)")
print(f"平均奖励: {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")
print(f"中位数奖励: {np.median(rewards):.4f}")

# 分阶段分析
early_rewards = rewards[:20]
mid_rewards = rewards[20:60]
late_rewards = rewards[60:]
print(f"\n分阶段分析:")
print(f"  早期 (1-20轮):   {np.mean(early_rewards):.4f} ± {np.std(early_rewards):.4f}")
print(f"  中期 (21-60轮):  {np.mean(mid_rewards):.4f} ± {np.std(mid_rewards):.4f}")
print(f"  后期 (61-100轮): {np.mean(late_rewards):.4f} ± {np.std(late_rewards):.4f}")

# 2. 策略偏好分析
preferences = np.array([h['preference'] for h in history])
print("\n" + "=" * 80)
print("2. 策略偏好演化分析")
print("=" * 80)
print(f"初始偏好: Efficiency={preferences[0][0]:.3f}, Cost={preferences[0][1]:.3f}, Delivery={preferences[0][2]:.3f}")
print(f"最终偏好: Efficiency={preferences[-1][0]:.3f}, Cost={preferences[-1][1]:.3f}, Delivery={preferences[-1][2]:.3f}")

# 后20轮的稳定偏好
stable_pref = preferences[-20:].mean(axis=0)
stable_std = preferences[-20:].std(axis=0)
print(f"\n稳定偏好 (后20轮平均):")
print(f"  Efficiency: {stable_pref[0]:.3f} ± {stable_std[0]:.3f} ({stable_pref[0]*100:.1f}%)")
print(f"  Cost:       {stable_pref[1]:.3f} ± {stable_std[1]:.3f} ({stable_pref[1]*100:.1f}%)")
print(f"  Delivery:   {stable_pref[2]:.3f} ± {stable_std[2]:.3f} ({stable_pref[2]*100:.1f}%)")

# 偏好变化率
pref_changes = np.linalg.norm(np.diff(preferences, axis=0), axis=1)
print(f"\n偏好变化趋势:")
print(f"  早期平均变化 (1-20轮):   {np.mean(pref_changes[:20]):.4f}")
print(f"  中期平均变化 (20-60轮):  {np.mean(pref_changes[20:60]):.4f}")
print(f"  后期平均变化 (60-100轮): {np.mean(pref_changes[60:]):.4f}")

# 3. 智能体学习分析
print("\n" + "=" * 80)
print("3. 各智能体学习曲线分析")
print("=" * 80)

for agent_id in ['agent0', 'agent1', 'agent2']:
    agent_name = ['作业选择智能体', '机器选择智能体', '运输机器人智能体'][int(agent_id[-1])]
    print(f"\n{agent_name} (policy_{agent_id}):")
    
    policy_losses = [h['agent_metrics'][agent_id]['policy_loss'] for h in history]
    vf_losses = [h['agent_metrics'][agent_id]['vf_loss'] for h in history]
    entropies = [h['agent_metrics'][agent_id]['entropy'] for h in history]
    
    print(f"  策略损失:")
    print(f"    初始: {policy_losses[0]:.6f}")
    print(f"    最终: {policy_losses[-1]:.6f}")
    print(f"    平均: {np.mean(policy_losses):.6f} ± {np.std(policy_losses):.6f}")
    
    print(f"  价值函数损失:")
    print(f"    初始: {vf_losses[0]:.4f}")
    print(f"    最终: {vf_losses[-1]:.4f}")
    print(f"    降低: {vf_losses[0] - vf_losses[-1]:.4f} ({(vf_losses[0] - vf_losses[-1])/vf_losses[0]*100:.1f}%)")
    
    print(f"  熵值 (探索度):")
    print(f"    初始: {entropies[0]:.4f}")
    print(f"    最终: {entropies[-1]:.4f}")
    print(f"    降低: {entropies[0] - entropies[-1]:.4f} ({(entropies[0] - entropies[-1])/entropies[0]*100:.1f}%)")

# 4. 探索-利用平衡
epsilons = [h['exploration_epsilon'] for h in history]
print("\n" + "=" * 80)
print("4. 探索-利用平衡")
print("=" * 80)
print(f"初始 epsilon: {epsilons[0]:.3f}")
print(f"最终 epsilon: {epsilons[-1]:.3f}")
print(f"衰减速率: {(epsilons[0] - epsilons[-1]) / len(epsilons):.6f} 每轮")

# 5. 可视化
fig, axes = plt.subplots(3, 2, figsize=(15, 12))

# 5.1 奖励曲线
ax = axes[0, 0]
ax.plot(range(1, len(rewards) + 1), rewards, 'b-', alpha=0.6, label='Episode Reward')
# 添加滑动平均
window = 10
ma_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
ax.plot(range(window, len(rewards) + 1), ma_rewards, 'r-', linewidth=2, label=f'{window}-Episode MA')
ax.axhline(y=max(rewards), color='g', linestyle='--', alpha=0.5, label=f'Best: {max(rewards):.2f}')
ax.set_xlabel('Iteration')
ax.set_ylabel('Reward')
ax.set_title('训练奖励曲线')
ax.legend()
ax.grid(True, alpha=0.3)

# 5.2 偏好演化
ax = axes[0, 1]
ax.plot(range(1, len(preferences) + 1), preferences[:, 0], label='Efficiency', linewidth=2)
ax.plot(range(1, len(preferences) + 1), preferences[:, 1], label='Cost', linewidth=2)
ax.plot(range(1, len(preferences) + 1), preferences[:, 2], label='Delivery', linewidth=2)
ax.set_xlabel('Iteration')
ax.set_ylabel('Preference Weight')
ax.set_title('策略偏好演化')
ax.legend()
ax.grid(True, alpha=0.3)

# 5.3 价值函数损失
ax = axes[1, 0]
for agent_id, name, color in [('agent0', 'Job Agent', 'blue'), 
                                ('agent1', 'Machine Agent', 'green'),
                                ('agent2', 'Transbot Agent', 'red')]:
    vf_losses = [h['agent_metrics'][agent_id]['vf_loss'] for h in history]
    ax.plot(range(1, len(vf_losses) + 1), vf_losses, color=color, label=name, alpha=0.7)
ax.set_xlabel('Iteration')
ax.set_ylabel('Value Function Loss')
ax.set_title('价值函数损失收敛')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# 5.4 熵值变化
ax = axes[1, 1]
for agent_id, name, color in [('agent0', 'Job Agent', 'blue'), 
                                ('agent1', 'Machine Agent', 'green'),
                                ('agent2', 'Transbot Agent', 'red')]:
    entropies = [h['agent_metrics'][agent_id]['entropy'] for h in history]
    ax.plot(range(1, len(entropies) + 1), entropies, color=color, label=name, alpha=0.7)
ax.set_xlabel('Iteration')
ax.set_ylabel('Entropy')
ax.set_title('策略熵值 (探索度)')
ax.legend()
ax.grid(True, alpha=0.3)

# 5.5 偏好变化率
ax = axes[2, 0]
ax.plot(range(1, len(pref_changes) + 1), pref_changes, 'purple', alpha=0.6)
# 添加滑动平均
ma_changes = np.convolve(pref_changes, np.ones(window)/window, mode='valid')
ax.plot(range(window, len(pref_changes) + 1), ma_changes, 'orange', linewidth=2, label=f'{window}-Iter MA')
ax.set_xlabel('Iteration')
ax.set_ylabel('Preference Change (L2 norm)')
ax.set_title('偏好变化率 (策略稳定性指标)')
ax.legend()
ax.grid(True, alpha=0.3)

# 5.6 探索率衰减
ax = axes[2, 1]
ax.plot(range(1, len(epsilons) + 1), epsilons, 'brown', linewidth=2)
ax.set_xlabel('Iteration')
ax.set_ylabel('Exploration Epsilon')
ax.set_title('探索率衰减曲线')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_dir / 'detailed_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n✅ 详细可视化已保存至: {results_dir / 'detailed_analysis.png'}")

# 6. 关键发现总结
print("\n" + "=" * 80)
print("关键发现总结")
print("=" * 80)

print("\n【训练稳定性】")
if rewards[-1] > rewards[0]:
    print(f"✅ 训练收敛良好,奖励提升 {abs(rewards[-1] - rewards[0]):.4f}")
else:
    print(f"⚠️  奖励略有下降 {abs(rewards[-1] - rewards[0]):.4f},可能需要调整超参数")

print("\n【策略学习】")
if stable_std.max() < 0.1:
    print(f"✅ 策略偏好已收敛 (最大标准差: {stable_std.max():.4f})")
else:
    print(f"⚠️  策略仍在探索中 (最大标准差: {stable_std.max():.4f})")

dominant_obj = ['效率', '成本', '交期'][np.argmax(stable_pref)]
print(f"✅ 策略倾向优化: {dominant_obj} ({stable_pref.max()*100:.1f}%)")

print("\n【智能体收敛】")
for agent_id in ['agent0', 'agent1', 'agent2']:
    vf_losses = [h['agent_metrics'][agent_id]['vf_loss'] for h in history]
    improvement = (vf_losses[0] - vf_losses[-1]) / vf_losses[0] * 100
    agent_name = ['作业', '机器', '运输'][int(agent_id[-1])]
    if improvement > 50:
        print(f"✅ {agent_name}智能体: 价值函数损失降低 {improvement:.1f}% (优秀)")
    elif improvement > 20:
        print(f"✓ {agent_name}智能体: 价值函数损失降低 {improvement:.1f}% (良好)")
    else:
        print(f"⚠️  {agent_name}智能体: 价值函数损失降低 {improvement:.1f}% (需改进)")

print("\n【探索-利用平衡】")
if epsilons[-1] <= 0.05:
    print(f"✅ 已切换至利用阶段 (epsilon={epsilons[-1]:.3f})")
else:
    print(f"⚠️  仍在探索阶段 (epsilon={epsilons[-1]:.3f})")

print("\n【后续建议】")
print("1. 如需进一步优化,可考虑:")
print("   - 增加训练轮数至 200 轮")
print("   - 调整学习率衰减策略")
print("   - 微调策略网络架构")
print("2. 当前模型已可用于:")
print("   - 基准测试对比")
print("   - 规则方法比较")
print("   - 实际问题验证")

print("\n" + "=" * 80)
