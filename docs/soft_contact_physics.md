# 柔性接触物理参数设置

## 概述

为了实现物体之间几乎无刚性碰撞的柔性接触效果，我们启用了PyBullet的柔性接触模型（Soft Contact Model），使物体碰撞时只是轻轻接触而不会发生强烈的运动反应。

## 核心参数说明

### 1. **contactStiffness（接触刚度）**
```python
contactStiffness=1000.0  # 柔性接触刚度
```
- **作用**：控制接触点的"弹簧"刚度
- **效果**：
  - 值**越小**：接触越柔软，物体会有轻微的"压陷"效果
  - 值**越大**：接触越硬，趋近于刚性接触
- **推荐范围**：
  - 极柔软：`100-500`
  - 中等柔软：`500-2000`（当前：1000）
  - 较硬：`2000-10000`

### 2. **contactDamping（接触阻尼）**
```python
contactDamping=100.0  # 柔性接触阻尼
```
- **作用**：抑制接触振动和反弹
- **效果**：
  - 值**越小**：碰撞后会有振荡
  - 值**越大**：碰撞后快速稳定，几乎无振荡
- **推荐范围**：
  - 低阻尼：`10-50`
  - 中等阻尼：`50-200`（当前：100）
  - 高阻尼：`200-1000`

### 3. **contactProcessingThreshold（接触处理阈值）**
```python
contactProcessingThreshold=0.01  # 从0.001增大到0.01
```
- **作用**：控制何时认为两个物体"接触"
- **效果**：
  - 值**越小**：接触检测更精确，但碰撞更"硬"
  - 值**越大**：接触检测更宽松，碰撞更"软"
- **推荐范围**：
  - 精确接触：`0.0-0.001`
  - 柔性接触：`0.01-0.1`（当前：0.01）

### 4. **restitution（弹性系数）**
```python
restitution=0.0  # 完全无弹性
```
- **作用**：控制碰撞后的反弹程度
- **效果**：
  - `0.0`：完全无反弹（当前设置）
  - `1.0`：完全弹性反弹
- **建议**：保持 `0.0` 以避免任何反弹

### 5. **线性和角度阻尼**
```python
linearDamping=0.95   # 极高线性阻尼
angularDamping=0.95  # 极高角度阻尼
```
- **作用**：让物体运动快速衰减
- **效果**：物体在碰撞后几乎立即停止
- **范围**：`0.0-1.0`，当前已接近最大值

## 物理行为对比

### 刚性接触（默认）
```python
# PyBullet 默认设置
# contactStiffness 和 contactDamping 未设置
contactProcessingThreshold=0.0
restitution=0.5
```
**表现**：
- ❌ 物体碰撞时产生强烈的力
- ❌ 可能发生明显的反弹和位移
- ❌ 精确堆叠困难

### 柔性接触（当前设置）
```python
contactStiffness=1000.0
contactDamping=100.0
contactProcessingThreshold=0.01
restitution=0.0
linearDamping=0.95
angularDamping=0.95
```
**表现**：
- ✅ 物体碰撞时几乎无强烈反应
- ✅ 接触时轻轻"贴合"，无明显位移
- ✅ 适合精确堆叠和放置操作

## 参数调节指南

### 如果物体还是有轻微移动：

**方案1：进一步软化接触**
```python
contactStiffness=500.0        # 降低刚度（从1000降到500）
contactDamping=200.0          # 增加阻尼（从100增到200）
contactProcessingThreshold=0.05  # 增大阈值（从0.01到0.05）
```

**方案2：降低质量**
```python
# 在 simple_stacking.py 中
mass = 0.1  # 从0.5降低到0.1
```

**方案3：增加摩擦力**
```python
lateralFriction=5.0  # 从2.5增加到5.0
```

### 如果物体"陷入"地面或容器：

**方案：增加刚度**
```python
contactStiffness=5000.0  # 从1000增加到5000
```

### 如果需要完全静态（无任何运动）：

**方案：使用临时固定**
```python
# 在放置后立即固定物体
p.changeDynamics(object_id, -1, mass=0.0)
```

## 实现位置

以下文件已更新：

1. **`src/phyvpuzzle/environment/base_env.py`**
   - `add_object()` 方法（第701-713行）：URDF物体的柔性接触设置
   - `create_primitive_object()` 方法（第806-818行）：基础形状物体的柔性接触设置

## 测试验证

运行测试任务查看效果：

```bash
cd /mnt/moonfs/wuyuhao-m2/wyh/VisualReasonBench
python examples/puzzle_translater.py
```

**观察要点**：
1. 物体移动到容器时是否轻轻落下（无强烈碰撞）
2. 堆叠物体时下层物体是否保持稳定（无位移）
3. 整体场景是否稳定（无抖动）

## 高级：自定义配置

如果需要为不同物体设置不同的柔性参数，可以在配置文件中添加：

```yaml
environment:
  soft_contact_config:
    stiffness: 1000.0
    damping: 100.0
    threshold: 0.01
```

然后在代码中读取：

```python
stiffness = getattr(self.config, "soft_contact_config", {}).get("stiffness", 1000.0)
damping = getattr(self.config, "soft_contact_config", {}).get("damping", 100.0)
```

## 参考资料

- [PyBullet Physics Parameters](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.9i02ojf4k3ve)
- PyBullet `changeDynamics` API: 控制物体动力学属性
- Contact Stiffness & Damping: 实现柔性接触的关键参数

