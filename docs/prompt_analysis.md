# Prompt 信息传递分析

## 📊 当前系统的信息传递流程

### 1. 静态信息（任务开始时）

通过 `_get_initial_system_prompt()` 和 `_get_initial_instruction()` 方法提供：

**three_by_three_stacking.py:**
```python
def _get_initial_instruction(self) -> str:
    num_pieces = len(self.environment.objects) - 1
    return f"""3D CUBE STACKING PUZZLE

You have {num_pieces} 3D puzzle pieces and one container.

TASK:
Assemble all pieces into the container to form a solid 3×3×3 cube (27 unit cubes total).
```

**simple_stacking.py:**
```python
def _get_initial_instruction(self) -> str:
    return """简单堆叠任务 - Puzzle Translater 测试

场景说明：
- 你面前有1个黑色容器（container）和2个拼图块
- Object #7 (ID: 2) - 第一个需要移动的拼图块
- Object #6 (ID: 3) - 第二个需要移动的拼图块
```

### 2. 动态信息（每步迭代更新）

通过 `get_object_mapping()` 方法在 `runner.py` 中提供：

**当前实现（runner.py，第267-317行）：**
```python
def get_object_mapping(self) -> str:
    """返回 object_id 到视觉属性（颜色、名称、类型）的映射"""
    
    lines = ["OBJECT MAPPING (object_id → properties):"]
    lines.append("=" * 60)
    
    for obj_info in self.environment.objects:
        if obj_info.properties.get('is_container', False):
            continue
        
        obj_id = obj_info.object_id
        
        # 获取颜色信息
        rgba_color = visual_shapes[0][7]
        r = int(rgba_color[0] * 255)
        g = int(rgba_color[1] * 255)
        b = int(rgba_color[2] * 255)
        
        lines.append(f"object_id={obj_id}, RGB=({r}, {g}, {b})")
    
    return "\n".join(lines)
```

**输出示例：**
```
OBJECT MAPPING (object_id → properties):
============================================================
object_id=2, RGB=(255, 0, 0)
object_id=3, RGB=(0, 0, 255)
============================================================
Total movable objects: 2
```

## ❌ 当前的问题

### 问题 1：缺少位置信息

`get_object_mapping()` **只提供了颜色信息**，没有提供：
- ✗ 物体的当前位置 (position)
- ✗ 物体的方向 (orientation)  
- ✗ 物体的名称 (name)
- ✗ 物体与容器的相对位置

### 问题 2：信息不一致

- `three_by_three_stacking`: 英文prompt，通用描述
- `simple_stacking`: 中文prompt，具体描述，但提到了"Object #7 (ID: 2)"这样的标识

**但实际上：**
- "Object #7" 是模型文件名（obj_7.urdf）
- "ID: 2" 是 object_id（PyBullet 的唯一标识符）
- 这些信息并没有在 `get_object_mapping()` 中提供！

### 问题 3：没有随环境迭代更新

虽然 `get_object_mapping()` 在每次 `_build_prompt_history()` 时都会被调用，但是：
- ObjectInfo 中**有** position 和 orientation 信息
- 但 `get_object_mapping()` **没有使用**这些信息

## ✅ 解决方案

### 方案 1：增强 `get_object_mapping()` 方法

在 `runner.py` 中修改，添加位置信息：

```python
def get_object_mapping(self) -> str:
    """返回完整的物体信息，包括位置、颜色、名称等"""
    
    lines = ["🧩 OBJECT MAPPING (完整物体信息):"]
    lines.append("=" * 80)
    
    non_container_count = 0
    
    for obj_info in self.environment.objects:
        # 处理容器
        if obj_info.properties.get('is_container', False):
            pos = obj_info.position
            lines.append(f"📦 Container:")
            lines.append(f"   - object_id: {obj_info.object_id}")
            lines.append(f"   - name: {obj_info.name}")
            lines.append(f"   - position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
            lines.append("")
            continue
        
        non_container_count += 1
        obj_id = obj_info.object_id
        pos = obj_info.position
        
        # 获取颜色
        try:
            visual_shapes = p.getVisualShapeData(obj_id)
            if visual_shapes:
                rgba_color = visual_shapes[0][7]
                r = int(rgba_color[0] * 255)
                g = int(rgba_color[1] * 255)
                b = int(rgba_color[2] * 255)
                color_str = f"RGB=({r}, {g}, {b})"
            else:
                color_str = "color=unknown"
        except:
            color_str = "color=error"
        
        # 格式化输出
        lines.append(f"🧩 Object #{non_container_count} (object_id: {obj_id}):")
        lines.append(f"   - name: {obj_info.name}")
        lines.append(f"   - color: {color_str}")
        lines.append(f"   - position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        lines.append(f"   - properties: {obj_info.properties}")
        lines.append("")
    
    lines.append("=" * 80)
    lines.append(f"Total movable objects: {non_container_count}")
    lines.append("\n💡 Use object_id (integer) to interact with objects via tool calls.")
    lines.append("   Position format: (x, y, z) in meters, z is height.")
    
    return "\n".join(lines)
```

**增强后的输出示例：**
```
🧩 OBJECT MAPPING (完整物体信息):
================================================================================
📦 Container:
   - object_id: 1
   - name: container
   - position: (-0.300, 0.000, 0.050)

🧩 Object #1 (object_id: 2):
   - name: piece_2
   - color: RGB=(255, 0, 0)
   - position: (0.150, -0.100, 0.100)
   - properties: {'index': 2, 'is_container': False, 'target_order': 1}

🧩 Object #2 (object_id: 3):
   - name: piece_3
   - color: RGB=(0, 0, 255)
   - position: (0.150, 0.100, 0.100)
   - properties: {'index': 3, 'is_container': False, 'target_order': 2}

================================================================================
Total movable objects: 2

💡 Use object_id (integer) to interact with objects via tool calls.
   Position format: (x, y, z) in meters, z is height.
```

### 方案 2：统一任务描述

让 `simple_stacking` 和 `three_by_three_stacking` 使用一致的物体引用方式：

**修改 simple_stacking.py:**

```python
def _get_initial_instruction(self) -> str:
    return """简单堆叠任务 - Puzzle Translater 测试

场景说明：
- 你面前有1个黑色容器（container）和2个拼图块
- 所有物体的详细信息（ID、位置、颜色）会在下方的 OBJECT MAPPING 中提供

任务步骤：
1. 将红色拼图块（piece_2）移动到黑色容器内的右上角位置
2. 将蓝色拼图块（piece_3）放置在 piece_2 的正上方，完全对齐

目标要求：
- 两个拼图块都必须完全位于容器边界内
- piece_3 必须在 piece_2 的上方（形成垂直堆叠）
- 确保堆叠稳定，不会倒塌

操作规则：
- 使用 object_id 来操作物体（不要使用名称）
- 你一次只能移动或旋转一个拼图块
- 参考 OBJECT MAPPING 中的实时位置信息来规划动作
- 按顺序完成两个步骤，直到任务完成

这是一个简单的测试任务，用于验证基本的物体操作和堆叠能力。
"""
```

## 📈 改进效果对比

### 改进前

**Agent收到的信息：**
- ✓ 任务描述（静态）
- ✓ 物体颜色（RGB值）
- ✗ 物体位置
- ✗ 物体名称
- ✗ 物体与容器的关系

### 改进后

**Agent收到的信息：**
- ✓ 任务描述（静态）
- ✓ 物体颜色（RGB值）
- ✓ 物体实时位置（每步更新）
- ✓ 物体名称
- ✓ 物体与容器的相对位置
- ✓ 物体的 properties（如 target_order）

## 🎯 实现建议

1. **立即修改** `runner.py` 的 `get_object_mapping()` 方法以包含位置信息
2. **更新** `simple_stacking.py` 的指令，使用更通用的物体引用方式
3. **保持一致** 两个任务的prompt风格（或都用英文，或都用中文）
4. **测试验证** 修改后agent是否能更好地理解物体位置关系

## 💡 额外建议

可以考虑添加相对位置信息：

```python
# 在 get_object_mapping() 中添加
lines.append("\n📍 SPATIAL RELATIONSHIPS:")
lines.append("=" * 80)

# 计算每个piece与container的相对位置
for obj_info in self.environment.objects:
    if not obj_info.properties.get('is_container', False):
        # 计算距离和方向
        distance = calculate_distance(obj_info.position, container_position)
        direction = calculate_direction(obj_info.position, container_position)
        lines.append(f"- {obj_info.name} is {distance:.3f}m away from container, direction: {direction}")
```

这样可以帮助agent更好地理解空间关系。

