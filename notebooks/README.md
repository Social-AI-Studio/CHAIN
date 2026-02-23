# Puzzle Interactive Jupyter Notebook 使用说明

## 概述
我已经创建了一个详细的Jupyter notebook (`puzzle_interactive.ipynb`)，让你可以在Jupyter环境中直接与puzzle环境进行交互。

## 重要修复
✅ **已修复路径问题**：notebook现在会正确加载3x3-stacking-puzzle模型，而不是使用简单的替代模型。

关键修复在于设置正确的URDF路径：
```python
urdf_path = project_root / "src" / "phyvpuzzle" / "environment" / "phobos_models"
```

## 主要文件

1. **puzzle_interactive.ipynb** - 主Jupyter notebook
   - 完整的交互式环境
   - 可视化功能
   - 动作执行和图像保存

2. **puzzle_interactive_test.py** - Python测试脚本
   - 用于独立测试功能

3. **verify_model_path.py** - 路径验证脚本
   - 确认3x3模型路径正确

## 使用方法

### 1. 启动Jupyter Notebook
```bash
cd /mnt/moonfs/wuyuhao-m2/wyh/VisualReasonBench/notebooks
jupyter notebook puzzle_interactive.ipynb
```

### 2. 按顺序运行单元格

notebook包含以下部分：
- 导入依赖项
- 定义InteractivePuzzleEnvironment类
- 初始化环境
- 探索环境（显示对象映射、可用工具等）
- 交互式动作执行
- 结果保存和分析

### 3. 核心功能

#### 环境管理
- `env.setup()` - 设置环境和任务
- `env.reset()` - 重置环境
- `env.close()` - 关闭环境

#### 信息获取
- `env.get_object_mapping()` - 显示对象ID和颜色映射
- `env.print_available_tools()` - 显示可用工具
- `env.print_task_info()` - 显示任务信息

#### 动作执行
- `env.execute_action(action_type, parameters)` - 执行动作

可用动作：
- `move_object` - 移动对象到指定位置
- `rotate_object` - 旋转对象
- `observe` - 改变观察角度
- `place_into_container` - 将拼图块放入容器
- `finish` - 完成任务

#### 可视化
- `env.display_image()` - 显示当前图像
- `env.save_image(filename)` - 保存当前图像
- `env.create_animation()` - 创建动画GIF

#### 数据保存
- `env.save_trajectory()` - 保存交互轨迹

## 示例代码

```python
# 初始化环境
env = InteractivePuzzleEnvironment()
env.setup()

# 显示对象信息
print(env.get_object_mapping())

# 执行动作
obs = env.execute_action("move_object", {
    "object_id": 1,
    "position": [0.0, 0.0, 0.1]
})

# 显示结果
env.display_image()

# 保存图像
env.save_image("step_1.png")
```

## 验证3x3模型加载

运行以下命令验证路径设置：
```bash
python verify_model_path.py
```

应该看到：
```
✅ Success! The 3x3-stacking-puzzle model directory found.
Contents:
  - obj_1
  - obj_2
  - obj_3
  - obj_4
  - obj_5
  - obj_6
  - obj_7
  - obj_8
```

## 注意事项

1. 确保在正确的目录下运行：`/mnt/moonfs/wuyuhao-m2/wyh/VisualReasonBench/notebooks`
2. 第一次运行可能需要一些时间来初始化PyBullet环境
3. 最大步数限制为10步（可在配置中修改）
4. 所有图像会自动保存到当前目录

## 故障排除

如果遇到问题：
1. 确保路径设置正确
2. 检查PyBullet是否正确安装
3. 如果环境无响应，使用`env.close()`关闭后重新初始化

## 完成的功能

✅ 从puzzle_quick.py提取环境加载和agent交互代码
✅ 创建交互式Jupyter notebook
✅ 实现可视化和PNG保存功能
✅ 修复路径问题，确保加载正确的3x3-stacking-puzzle模型
✅ 提供完整的API和使用示例

notebook现在可以正确加载和使用原始的3x3-stacking-puzzle模型了！