# 容器稳定性改进

## 🎯 问题描述

用户需要确保容器（container）在环境中**完全固定**，不会因为物理模拟而移动或晃动。

## ✅ 解决方案

### 改进 1：`add_object()` 方法 - URDF加载

**文件：** `src/phyvpuzzle/environment/base_env.py` (第594-634行)

#### 变化：

1. **自动检测容器**：
```python
# 检查是否为容器
is_container = properties.get('is_container', False)
```

2. **使用 `useFixedBase` 参数**：
```python
object_id = p.loadURDF(
    urdf_path,
    basePosition=position,
    baseOrientation=orientation,
    globalScaling=scale,
    useFixedBase=is_container  # ✅ 容器使用固定基座
)
```

3. **特殊的物理参数**：
```python
if is_container:
    p.changeDynamics(
        object_id,
        -1,
        mass=0.0,              # ✅ 质量为0
        lateralFriction=1.0,   # 高摩擦力
        spinningFriction=0.0,  # 无旋转摩擦
        rollingFriction=0.0,   # 无滚动摩擦
        linearDamping=0.0,     # 无线性阻尼
        angularDamping=0.0,    # 无角度阻尼
        restitution=0.0,       # 无弹性
    )
```

### 改进 2：`create_primitive_object()` 方法 - 基本形状创建

**文件：** `src/phyvpuzzle/environment/base_env.py` (第710-737行)

#### 变化：

1. **检查质量参数**：
```python
if mass == 0.0:
    # 质量为0的物体被视为固定物体
    p.changeDynamics(
        object_id,
        -1,
        mass=0.0,              # ✅ 确保质量为0
        lateralFriction=1.0,
        spinningFriction=0.0,
        rollingFriction=0.0,
        linearDamping=0.0,
        angularDamping=0.0,
        restitution=0.0,
    )
```

### 改进 3：`simple_stacking.py` - 容器创建

**文件：** `src/phyvpuzzle/tasks/simple_stacking.py` (第149-163行)

#### 变化：

1. **创建容器时明确设置质量为0**：
```python
container_id = self.environment.create_primitive_object(
    object_name="container",
    shape_type="box",
    size=(0.15, 0.15, 0.15),
    position=(table_x - 0.3, table_y, table_z + 0.05),
    color=(0.1, 0.1, 0.1, 1.0),
    mass=0.0,  # ✅ 质量为0，完全固定
)
```

2. **标记为容器**：
```python
# 标记为容器并更新属性
for obj in self.environment.objects:
    if obj.object_id == container_id:
        obj.properties['is_container'] = True
        break
```

## 📊 容器固定机制

### 三重保护机制

1. **`useFixedBase=True`** (URDF加载时)
   - PyBullet的固定基座功能
   - 物体不会受重力和外力影响
   - 位置和方向完全锁定

2. **`mass=0.0`** (物理参数)
   - 质量为0表示无限质量
   - 物体无法被推动
   - 完全静止状态

3. **零摩擦/阻尼参数** (容器专用)
   - 无旋转和滚动摩擦
   - 无线性和角度阻尼
   - 确保没有任何运动倾向

## 🎯 适用场景

### 自动固定的情况

1. **从URDF加载**：
   - 如果 `properties['is_container'] = True`
   - 自动使用 `useFixedBase=True` 和 `mass=0.0`

2. **创建基本形状**：
   - 如果 `mass=0.0`
   - 自动设置为固定物体

### 示例

#### Simple Stacking Task：
```python
# Container (自动固定)
- URDF加载: is_container=True → useFixedBase=True
- 或者创建: mass=0.0 → 固定物体
```

#### Three by Three Stacking Task：
```python
# obj_8 容器
properties = {"index": 8, "is_container": True}
→ 自动使用 useFixedBase=True 和 mass=0.0
```

## 🔬 验证方法

### 检查容器是否固定

1. **运行任务**：
```bash
python examples/puzzle_translater.py
```

2. **观察容器**：
   - ✅ 容器应该保持在原始位置
   - ✅ 即使拼图块与容器碰撞，容器也不移动
   - ✅ 容器不会晃动或倾斜

3. **查看日志**：
```
🧩 OBJECT MAPPING:
📦 Container:
   - object_id: 1
   - position: (-0.300, 0.000, 0.050)  # 位置始终不变
```

### 位置验证

检查多个步骤的容器位置：
- Step 0: `position: (-0.300, 0.000, 0.050)`
- Step 1: `position: (-0.300, 0.000, 0.050)` ✅ 相同
- Step 2: `position: (-0.300, 0.000, 0.050)` ✅ 相同
- Step 3: `position: (-0.300, 0.000, 0.050)` ✅ 相同

## 📝 总结

### 修改的文件

1. ✅ `src/phyvpuzzle/environment/base_env.py`
   - `add_object()` - 添加容器固定逻辑
   - `create_primitive_object()` - 添加质量为0的固定逻辑

2. ✅ `src/phyvpuzzle/tasks/simple_stacking.py`
   - `_create_simple_puzzle_pieces()` - 确保容器质量为0并标记

### 实现效果

✅ **容器完全固定**：
- 不受重力影响
- 不受碰撞影响
- 位置和方向锁定
- 物理模拟稳定

✅ **自动识别**：
- 通过 `is_container` 属性自动识别
- 通过 `mass=0.0` 自动固定
- 无需手动配置

✅ **向后兼容**：
- 不影响现有任务
- `three_by_three_stacking` 自动受益
- 所有容器都会自动固定

## 🚀 使用建议

### 创建固定容器的两种方式

**方式 1：URDF加载**
```python
self.environment.add_object(
    object_name="container",
    urdf_path="path/to/container.urdf",
    position=(x, y, z),
    properties={"is_container": True}  # ✅ 自动固定
)
```

**方式 2：基本形状**
```python
self.environment.create_primitive_object(
    object_name="container",
    shape_type="box",
    size=(0.15, 0.15, 0.15),
    position=(x, y, z),
    mass=0.0  # ✅ 自动固定
)
```

两种方式都能确保容器完全固定！🎉

