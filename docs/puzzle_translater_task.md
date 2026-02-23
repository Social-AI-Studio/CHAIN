# Puzzle Translater 测试任务说明

## 📋 任务概述

**Puzzle Translater** 是一个简化的拼图堆叠任务，专门用于测试AI代理的基本物体操作和空间推理能力。相比完整的3×3×3立方体组装任务，这个任务更加简单直接。

## 🎯 任务目标

这个任务要求AI代理完成两个连续的物体操作：

1. **步骤1**：将 **Object #7 (ID: 2)** 移动到黑色容器（container）内
2. **步骤2**：将 **Object #6 (ID: 3)** 放置在 **Object #7 (ID: 2)** 的上方

### 成功标准

任务成功需要满足以下条件：
- ✅ Object #7 (ID: 2) 完全位于容器内部
- ✅ Object #6 (ID: 3) 在 Object #7 (ID: 2) 的上方（垂直堆叠）
- ✅ Object #6 (ID: 3) 也完全位于容器内部
- ✅ 两个块稳定堆叠，没有倒塌

## 🏗️ 任务实现

### 文件结构

```
VisualReasonBench/
├── eval_configs/
│   └── puzzle_translater.yaml          # 任务配置文件
├── examples/
│   └── puzzle_translater.py            # 运行脚本
└── src/phyvpuzzle/tasks/
    └── simple_stacking.py              # 任务实现类
```

### 核心组件

#### 1. 任务配置 (`puzzle_translater.yaml`)

```yaml
task:
  type: "simple_stacking"              # 任务类型
  name: "puzzle_translater_test"       # 任务名称
  difficulty: easy                      # 难度级别
  num_pieces: 2                        # 拼图块数量
  puzzle_size: [2, 1]                  # 简单堆叠
  piece_size: 0.08                     # 块大小
  ruled_evaluation: false              # 使用VLM评估
```

#### 2. 任务类 (`simple_stacking.py`)

任务类实现了以下关键方法：

- **`_get_initial_system_prompt()`**：定义AI代理的系统角色和行为准则
- **`_get_initial_instruction()`**：提供清晰的任务指令
- **`_load_puzzle_models()`**：加载3个对象（1个container + 2个拼图块）
- **`_evaluate_success()`**：评估任务是否成功完成

##### 系统Prompt（中文）

```
你是一个在基于物理的3D仿真环境中运行的智能AI代理。

角色定位：
- 你是一个空间推理和问题解决代理
- 你可以观察3D场景，理解物体几何，并规划物理交互
- 你通过逻辑性的、循序渐进的推理来做出决策

行为准则：
- 在操作物体时系统性地思考空间关系
- 使用观察结果来指导精确的、物理上有效的动作
- 在排列或组装物体时考虑稳定性、接触和适配
- 总是在行动前进行推理，并根据环境反馈调整计划
```

##### 任务指令（中文）

```
简单堆叠任务 - Puzzle Translater 测试

场景说明：
- 你面前有1个黑色容器（container）和2个拼图块
- Object #7 (ID: 2) - 第一个需要移动的拼图块
- Object #6 (ID: 3) - 第二个需要移动的拼图块

任务步骤：
1. 将 Object #7 (ID: 2) 移动到黑色容器内
2. 将 Object #6 (ID: 3) 放置在 Object #7 (ID: 2) 的上方

目标要求：
- 两个拼图块都必须完全位于容器边界内
- Object #6 必须在 Object #7 的上方（形成垂直堆叠）
- 确保堆叠稳定，不会倒塌
```

## 🚀 如何运行

### 方法1：直接运行脚本

```bash
cd /mnt/moonfs/wuyuhao-m2/wyh/VisualReasonBench
python examples/puzzle_translater.py
```

### 方法2：使用配置文件

```bash
cd /mnt/moonfs/wuyuhao-m2/wyh/VisualReasonBench
phyvpuzzle run --config eval_configs/puzzle_translater.yaml
```

## 📊 任务特点

| 特性 | puzzle_quick (3×3×3) | puzzle_translater |
|------|---------------------|-------------------|
| 任务复杂度 | 高 | 低 |
| 拼图块数量 | 9块 | 2块 |
| 最大步数 | 10步 | 5步 |
| 难度级别 | Medium | Easy |
| 最优步数 | 10步 | 2步 |
| 评估方式 | VLM判断 | VLM判断 |
| 主要挑战 | 3D空间规划 | 基本堆叠 |

## 🎓 设计理念

这个任务被设计为：

1. **简单明确**：只有两个步骤，目标清晰
2. **易于验证**：成功标准容易观察和判断
3. **循序渐进**：从简单任务开始，逐步提升难度
4. **教学友好**：适合用于演示和测试基本功能

## 📝 任务Prompt清晰度

✅ **Prompt完整性检查：**

- ✅ 系统角色定义清晰（`_get_initial_system_prompt`）
- ✅ 任务目标明确（`_get_initial_instruction`）
- ✅ 具体步骤详细说明
- ✅ 成功标准明确定义（`_evaluate_success_agent`）
- ✅ 使用中文说明，便于理解
- ✅ 包含场景描述和物体标识

## 🔍 与原任务对比

### puzzle_quick.yaml (原任务)

```yaml
task:
  type: "3x3_stacking"
  num_pieces: 9
  max_steps: 10
  difficulty: medium
```

**Prompt示例：**
- 目标：组装3×3×3立方体
- 挑战：复杂的空间规划

### puzzle_translater.yaml (新任务)

```yaml
task:
  type: "simple_stacking"
  num_pieces: 2
  max_steps: 5
  difficulty: easy
```

**Prompt示例：**
- 目标：堆叠2个特定的拼图块
- 挑战：基本的物体操作

## 🛠️ 扩展建议

如果想进一步定制任务，可以修改：

1. **拼图块数量**：在配置中修改 `num_pieces`
2. **难度级别**：调整 `difficulty` 和 `max_steps`
3. **任务指令**：在 `simple_stacking.py` 中修改 `_get_initial_instruction()`
4. **评估标准**：调整 `_evaluate_success_agent()` 中的成功标准

## 📚 相关文档

- [原始任务示例](../examples/puzzle_quick.py)
- [任务开发指南](../README_TASK_DEVELOPMENT.md)
- [系统配置说明](../docs/configuration.md)

## 💡 总结

**Puzzle Translater** 任务成功地创建了一个简化版的拼图任务，具有：

✅ **清晰的任务描述** - 通过系统prompt和任务指令明确说明  
✅ **简单的目标** - 只需要完成2个基本操作  
✅ **完整的实现** - 包含配置、任务类和运行脚本  
✅ **中文支持** - 所有说明和prompt都是中文  

这个任务非常适合用于快速测试和验证AI代理的基本物理推理能力！

