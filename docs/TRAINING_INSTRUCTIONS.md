# DFJSP-T 训练指南

## 重要说明：为什么使用 hierarchical_train_complete.py 而不是 dfjspt_train.py

### 问题根源

`DFJSPT/dfjspt_train.py` 原本设计用于**4-agent层级架构**（包括strategy作为RLlib agent），这种方法与RLlib 2.x存在不兼容问题：

- ❌ **"Single trajectory" 错误**：Strategy agent观察频率与tactical agents不一致
- ❌ **Episode边界冲突**：RLlib的multi-agent episode处理无法支持这种混合观察模式
- ❌ **复杂的配置**：需要同时管理4个agent的策略定义和映射

### Method 3 解决方案

经过多次调试和验证，**Method 3（策略元控制器）**是唯一成功的方案：

- ✅ **策略层外部化**：不作为RLlib agent，而是独立的PyTorch网络
- ✅ **Episode级操作**：在episode开始前设置偏好，episode结束后更新
- ✅ **清晰分离**：策略学习和战术训练完全独立
- ✅ **已验证成功**：50次迭代训练无错误完成

## 推荐使用方式

### ✅ 方法1：使用 hierarchical_train_complete.py（推荐）

这是专门为Method 3设计的简洁训练脚本：

```bash
conda activate dfjsp2t
python hierarchical_train_complete.py
```

**优点**：
- ✅ 专为Method 3设计
- ✅ 代码简洁清晰
- ✅ 已完整测试验证
- ✅ 包含完整的检查点和历史记录
- ✅ 无配置冲突

**配置**：
- 迭代次数：修改 `NUM_ITERATIONS = 50` （第36行）
- 检查点间隔：修改 `CHECKPOINT_INTERVAL = 10` （第37行）
- 策略更新频率：修改 `STRATEGY_UPDATE_FREQUENCY = 5` （第38行）

### ⚠️ 方法2：修改后的 dfjspt_train.py（不推荐）

虽然已经尝试修改 `DFJSPT/dfjspt_train.py` 来支持Method 3，但存在以下问题：

- ⚠️ Ray初始化在某些环境下不稳定
- ⚠️ 配置复杂，需要同时管理多个参数文件
- ⚠️ 包含大量遗留代码（为4-agent设计）
- ⚠️ 未经完整验证

**如果必须使用**：
```bash
# 1. 确保 use_tune = False in dfjspt_params.py
# 2. 运行
conda activate dfjsp2t
python DFJSPT/dfjspt_train.py
```

## 训练流程说明

### 使用 hierarchical_train_complete.py 的完整流程

#### 1. 启动训练

```bash
conda activate dfjsp2t
python hierarchical_train_complete.py
```

**预期输出**：
```
================================================================================
Hierarchical DFJSP-T Training (Method 3 - Strategy Meta-Controller)
================================================================================
Start time: 2025-10-16 XX:XX:XX
Problem size: 10J × 5M × 3T
Multi-objective: True
================================================================================
✓ Custom models registered
✓ Ray initialized
✓ Results will be saved to: DFJSPT/training_results/hierarchical_v3_YYYYMMDD_HHMMSS
✓ Algorithm built successfully
✓ Strategy controller created
```

#### 2. 训练进度监控

每次迭代会显示：
```
================================================================================
Iteration 1/50
================================================================================

🎯 Strategy Update:
  Preference vector: [0.xxx, 0.xxx, 0.xxx]
  Efficiency: 0.xxx
  Cost:       0.xxx
  Delivery:   0.xxx
  Exploration ε: 0.xxx

Training tactical agents...

📊 Results:
  Episode reward mean: -X.XX
  Episodes this iter:  XX
  Timesteps total:     XXXX
```

每10次迭代会保存检查点：
```
💾 Checkpoint saved:
  Location: DFJSPT/training_results/hierarchical_v3_YYYYMMDD_HHMMSS/checkpoint_XX
```

#### 3. 训练完成

```
================================================================================
Training Complete!
================================================================================
📦 Final checkpoint saved to: .../final_checkpoint
📊 Training history saved to: .../complete_training_history.json

================================================================================
Training Summary
================================================================================
Total iterations:     50
Total timesteps:      100000
Strategy buffer size: 50
Final preference:     [0.xxx, 0.xxx, 0.xxx]
  Efficiency: 0.xxx
  Cost:       0.xxx
  Delivery:   0.xxx
```

#### 4. 分析结果

```bash
python analyze_hierarchical_training.py
```

这会生成：
- 训练进度图表
- 偏好演化分析
- 学习稳定性报告
- 保存为 `training_analysis.png`

## 结果文件说明

训练结果保存在 `DFJSPT/training_results/hierarchical_v3_YYYYMMDD_HHMMSS/`：

```
hierarchical_v3_YYYYMMDD_HHMMSS/
├── checkpoint_10/
│   ├── algorithm_state.pkl          # RLlib tactical agents
│   ├── strategy_controller.pt       # Strategy network
│   └── training_history.json        # 训练记录
├── checkpoint_20/
├── checkpoint_30/
├── checkpoint_40/
├── checkpoint_50/
├── final_checkpoint/
│   ├── algorithm_state.pkl
│   └── strategy_controller.pt
├── complete_training_history.json   # 完整50次迭代历史
└── training_analysis.png            # 可视化分析图
```

## 核心配置参数

### hierarchical_train_complete.py 中的关键参数

```python
# 训练迭代数
NUM_ITERATIONS = 50

# 检查点保存间隔
CHECKPOINT_INTERVAL = 10

# 策略更新频率（每N个episode更新一次偏好）
STRATEGY_UPDATE_FREQUENCY = 5

# 探索率衰减
EXPLORATION_EPSILON_START = 0.3
EXPLORATION_EPSILON_END = 0.05
EXPLORATION_DECAY = 0.95

# RLlib训练参数
train_batch_size = 2000
sgd_minibatch_size = 256
num_sgd_iter = 10
lr = 3e-4
```

### dfjspt_params.py 中的相关参数

```python
# 问题规模
n_jobs = 10
n_machines = 5
n_transbots = 3

# 启用层级框架（Method 3）
use_hierarchical_framework = True

# 启用多目标奖励
use_multi_objective_reward = True

# 工作进程数
num_workers = 1
num_envs_per_worker = 1

# 使用Tune（仅用于原dfjspt_train.py）
use_tune = True  # hierarchical_train_complete.py不需要此参数
```

## 常见问题解答

### Q1: 为什么不直接修复 dfjspt_train.py？

**A**: `dfjspt_train.py` 为4-agent架构设计，代码逻辑复杂：
- 包含strategy agent的策略定义和映射
- 使用HierarchicalDfjsptEnv包装器（已废弃）
- 依赖于Tune框架的复杂配置
- 修改需要大量重构，风险高

`hierarchical_train_complete.py` 从零开始为Method 3设计，更清晰可靠。

### Q2: Method 3与原方案有什么不同？

**原方案（4-agent）**：
- Strategy作为RLlib的第4个agent
- 在每个step观察和决策
- 与RLlib的episode管理冲突 ❌

**Method 3（元控制器）**：
- Strategy是外部PyTorch网络
- Episode开始前设置偏好
- Episode结束后学习更新
- 与RLlib完全独立 ✅

### Q3: 训练时间需要多久？

根据已完成的训练：
- **50次迭代**：约3.6分钟
- **平均每次迭代**：约4.3秒
- **100,000 timesteps**：约3.6分钟

### Q4: 如何调整训练参数？

直接编辑 `hierarchical_train_complete.py`：

```python
# 修改这些值
NUM_ITERATIONS = 100        # 更多迭代
CHECKPOINT_INTERVAL = 20    # 更少检查点
STRATEGY_UPDATE_FREQUENCY = 10  # 更新频率
```

### Q5: 如何加载检查点继续训练？

目前需要手动实现。建议方式：

```python
# 在 hierarchical_train_complete.py 中
# 训练前加载
if os.path.exists("path/to/checkpoint"):
    algorithm.restore("path/to/checkpoint")
    strategy.load("path/to/strategy_controller.pt")
```

### Q6: 可以在原 dfjspt_train.py 上训练吗？

**不推荐**，但如果坚持：

1. 设置 `use_tune = False` in `dfjspt_params.py`
2. 确保 Ray 正确初始化
3. 注意可能的兼容性问题

**强烈建议**使用 `hierarchical_train_complete.py`。

## 架构对比总结

| 特性 | hierarchical_train_complete.py | dfjspt_train.py |
|------|--------------------------------|-----------------|
| **架构** | Method 3（元控制器） | 4-agent（已弃用） |
| **RLlib兼容性** | ✅ 完全兼容 | ❌ 有冲突 |
| **代码复杂度** | 简洁清晰 | 复杂遗留 |
| **验证状态** | ✅ 完整测试 | ⚠️ 部分修改 |
| **推荐使用** | ✅ **强烈推荐** | ❌ 不推荐 |

## 下一步建议

1. ✅ **使用 hierarchical_train_complete.py 进行生产训练**
2. 📊 **运行 analyze_hierarchical_training.py 分析结果**
3. 🔬 **对比基准性能**（运行标准3-agent训练）
4. 🎯 **调优超参数**（学习率、探索率等）
5. 📈 **扩展到更大规模问题**（20J×10M×5T等）

## 结论

**请使用 `hierarchical_train_complete.py`**，这是经过完整验证、专为Method 3设计的训练脚本。它简洁、可靠、已经在50次迭代中成功运行。

`dfjspt_train.py` 保留作为参考，但不推荐用于Method 3的层级训练。
