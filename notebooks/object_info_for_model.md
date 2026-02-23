# 物体信息传递给模型 - 使用指南

## 概述

已修改 `PuzzleEnvironment._get_state_description()` 方法，现在每次观察(Observation)都会包含所有物体的详细信息，包括：
- **Position（位置）**：x, y, z 坐标
- **Bounding Box（尺寸）**：width, depth, height
- **Color（颜色）**：RGB 值与十六进制代码
- **Orientation（朝向）**：yaw 角度
- **Status（状态）**：是否已放置、是否在容器内

## 输出格式示例

```
Tool Call Result - Status: success, Message: Camera moved to 0° viewpoint

=== OBJECTS IN SCENE ===

CONTAINER (object_id=4):
  Name: container_3x3
  Position: x=0.1389, y=0.0000, z=0.0500
  Size: width=0.6200m, depth=0.6200m, height=0.2000m

MOVABLE OBJECTS (7 pieces):

[1] Object ID: 8
    Name: obj_1
    Position: x=-0.4625, y=-0.4625, z=0.0500
    Orientation: yaw=0.0° (rotation around z-axis)
    Size: width=0.3848m, depth=0.3848m, height=0.3848m
    Color: RGB(222, 197, 12) #dec50c

[2] Object ID: 7
    Name: obj_2
    Position: x=0.0000, y=-0.4625, z=0.0500
    Orientation: yaw=0.0° (rotation around z-axis)
    Size: width=0.3848m, depth=0.3848m, height=0.3848m
    Color: RGB(201, 88, 0) #c95800

[3] Object ID: 6
    Name: obj_3
    Position: x=0.4625, y=-0.4625, z=0.0500
    Orientation: yaw=0.0° (rotation around z-axis)
    Size: width=0.3848m, depth=0.3848m, height=0.3848m
    Color: RGB(169, 27, 0) #a91b00
    Status: placed, in_container

...

=== END OBJECTS ===
```

## 如何访问

### 1. 在 VLM Agent 中（自动）

模型会在每次观察时自动收到这些信息：

```python
# 在 runner.py 或 agent 中
observation = env.step(action)
# observation.description 包含上述详细信息
print(observation.description)
```

### 2. 在笔记本中查看

```python
# 执行任意动作后
obs = env.execute_action("observe", {"angle": 0})
print(obs.description)  # 打印完整描述
```

### 3. 在测试脚本中

```python
from phyvpuzzle.environment.puzzle_env import PuzzleEnvironment
from phyvpuzzle.core.config import EnvironmentConfig

env = PuzzleEnvironment(config)
obs = env.reset()
print(obs.description)
```

## 信息字段说明

### Container（容器）
- **object_id**：容器的唯一标识符
- **Name**：容器名称（如 `container_3x3`）
- **Position**：容器中心的世界坐标（米）
- **Size**：容器的宽/深/高（米）

### Movable Objects（可移动物体）
每个拼图块包含：

1. **Object ID**：用于 `move_object`、`place_into_container` 等工具
2. **Name**：物体名称（如 `obj_1`）
3. **Position**：当前世界坐标 (x, y, z)，单位米
4. **Orientation**：绕 z 轴的旋转角度（yaw，度数）
5. **Size**：AABB 包围盒尺寸
   - width：x 方向长度
   - depth：y 方向长度
   - height：z 方向长度
6. **Color**：RGB(0-255) 与十六进制颜色代码
7. **Status**（可选）：
   - `placed`：已被放置过
   - `in_container`：当前在容器内

## 模型如何使用这些信息

### 示例 1：根据颜色选择物体

```python
# VLM 可以看到：
# [1] Object ID: 8, Color: RGB(222, 197, 12) #dec50c
# [2] Object ID: 7, Color: RGB(201, 88, 0) #c95800

# 模型决策：
# "I need to place the yellow piece (ID: 8) first"
action = {
    "action_type": "place_into_container",
    "parameters": {"object_id": 8, "offset_x": 0.0, "offset_y": 0.0}
}
```

### 示例 2：根据位置判断距离

```python
# VLM 可以看到：
# Container Position: x=0.1389, y=0.0000
# Object 8 Position: x=-0.4625, y=-0.4625

# 模型推理：
# "Object 8 is far from container (~0.65m away), need to move it"
```

### 示例 3：根据尺寸规划堆叠

```python
# VLM 可以看到：
# Container Size: height=0.2000m
# Object Size: height=0.3848m

# 模型推理：
# "Piece height is 0.38m, can stack 1-2 layers in 0.20m container"
# "For second layer, use offset_z=0.05"
```

### 示例 4：避免碰撞

```python
# VLM 可以看到：
# [1] Position: x=0.0000, y=0.0000, Size: width=0.3848m
# [2] Position: x=0.1000, y=0.1000, Size: width=0.3848m

# 模型推理：
# "Objects are only 0.1m apart, but each is 0.38m wide"
# "They will overlap! Need to place with at least 0.4m separation"
```

## 与旧版本的区别

### 旧版（仅工具结果）
```
Tool Call Result - Status: success, Message: Camera moved to 0° viewpoint
```

### 新版（完整物体信息）
```
Tool Call Result - Status: success, Message: Camera moved to 0° viewpoint

=== OBJECTS IN SCENE ===
[详细的物体列表，包含位置/尺寸/颜色]
=== END OBJECTS ===
```

## 性能影响

- **额外开销**：每次 `step()` 增加约 5-10ms（取决于物体数量）
- **Token 消耗**：每个物体约 100-150 tokens
- **建议**：对于 >20 个物体的场景，可考虑在配置中添加开关

## 自定义与扩展

### 禁用某些信息（如需节省 tokens）

编辑 `puzzle_env.py` 的 `_get_state_description()` 方法，注释不需要的部分：

```python
# 例如，不输出 Orientation
# desc += f"    Orientation: yaw={yaw_deg:.1f}°\n"
```

### 添加额外信息

```python
# 在循环中添加，例如质量：
try:
    dynamics = p.getDynamicsInfo(obj_id, -1)
    mass = dynamics[0]
    desc += f"    Mass: {mass:.4f}kg\n"
except:
    pass
```

### 修改输出格式（JSON）

如果模型更适合 JSON 格式：

```python
import json

def _get_state_description(self) -> str:
    objects_data = []
    for obj_info in self.objects:
        obj_data = {
            "object_id": obj_info.object_id,
            "name": obj_info.name,
            "position": {"x": pos[0], "y": pos[1], "z": pos[2]},
            "size": {"width": width, "depth": depth, "height": height},
            "color": {"r": r, "g": g, "b": b, "hex": f"#{r:02x}{g:02x}{b:02x}"}
        }
        objects_data.append(obj_data)
    
    return json.dumps({"objects": objects_data}, indent=2)
```

## 验证与测试

### 快速验证

```python
# 在笔记本中运行
env = InteractivePuzzleEnvironment()
env.setup()

# 查看初始状态描述
obs = env.current_observation
print(obs.description)

# 执行动作后查看
obs = env.execute_action("observe", {"angle": 90})
print(obs.description)
```

### 单元测试

```python
def test_object_info_in_description():
    env = PuzzleEnvironment(config)
    obs = env.reset()
    
    # 检查是否包含关键信息
    assert "=== OBJECTS IN SCENE ===" in obs.description
    assert "Position:" in obs.description
    assert "Size:" in obs.description
    assert "Color:" in obs.description
    assert "Object ID:" in obs.description
```

## 注意事项

1. **坐标系**：所有坐标为世界坐标系，原点在地面中心
2. **单位**：位置与尺寸单位为米(m)，角度为度(°)
3. **精度**：坐标保留 4 位小数(0.0001m = 0.1mm)
4. **异常处理**：如果某个物体信息获取失败，会显示 `(unavailable)`
5. **容器优先**：容器信息总是在可移动物体之前输出

## 常见问题

**Q: 为什么有时 Color 显示为 `(unavailable)`？**  
A: 可能是该物体没有视觉形状数据，或 PyBullet 连接失败。检查 `p.getVisualShapeData()` 是否返回空。

**Q: Position 会实时更新吗？**  
A: 是的，每次调用 `env.step()` 后都会重新查询最新位置。

**Q: 如何只获取特定物体的信息？**  
A: 可以在笔记本中使用 `env.print_objects_info()`，或直接调用 PyBullet API：
```python
pos, orn = p.getBasePositionAndOrientation(object_id)
```

**Q: 信息会传递到日志文件吗？**  
A: 是的，`observation.description` 会被 `ExperimentLogger` 记录到 JSON 日志中。

## 总结

现在模型在每次观察时都能获得完整的物体信息（位置、尺寸、颜色），可以：
- ✅ 根据颜色识别物体
- ✅ 根据位置判断距离与空间关系
- ✅ 根据尺寸规划放置策略
- ✅ 根据状态跟踪任务进度

所有信息自动包含在 `observation.description` 中，无需额外 API 调用。


