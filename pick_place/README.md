# 基于视觉的机械臂抓取系统

这是一个基于视觉反馈的机械臂抓取系统，能够根据相机检测到的物体距离和像素坐标，自动计算物体在机械臂基座坐标系下的位置，并执行精确的抓取任务。

## 功能特点

### 🎯 核心功能
- **精确的坐标转换**: 基于相机内参和手眼标定，将像素坐标转换为机械臂基座坐标
- **智能抓取序列**: 完整的10步抓取流程，包括安全检查和异常处理
- **灵活的视觉输入**: 支持距离信息和像素坐标的组合输入
- **实时位置计算**: 根据相机位姿动态计算物体位置

### 🔧 技术特点
- 使用手眼标定矩阵进行坐标系转换
- 支持相机内参自动加载
- 完整的异常处理和安全机制
- 详细的日志输出和调试信息

## 系统架构

### 核心组件
1. **VisionBasedPickPlace**: 主要的抓取系统类
2. **相机内参处理**: 自动加载和处理相机内参
3. **手眼标定**: 基于eye2tcp_matrix.txt的标定数据
4. **坐标转换**: 像素坐标→相机坐标→基座坐标

### 数学原理
基于相机内参的3D坐标计算：
```
x = (u - cx) × distance / fx
y = (v - cy) × distance / fy  
z = distance
```

## 快速开始

### 1. 基本使用
```python
from vision_based_pick_place import create_vision_pick_system, execute_vision_pick

# 创建系统
vision_system = create_vision_pick_system()

# 执行抓取
distance = 300.0  # mm
pixel_u = 700.0   # 像素坐标
pixel_v = 300.0   # 像素坐标

success = execute_vision_pick(vision_system, distance, pixel_u, pixel_v)
```

### 2. 仅计算物体位置
```python
from vision_based_pick_place import calculate_object_position

position = calculate_object_position(vision_system, distance, pixel_u, pixel_v)
print(f"物体位置: {position}")
```

### 3. 批量处理
```python
objects = [
    {"distance": 300.0, "u": 640.0, "v": 360.0, "name": "物体1"},
    {"distance": 280.0, "u": 720.0, "v": 300.0, "name": "物体2"}
]

for obj in objects:
    position = calculate_object_position(vision_system, obj["distance"], obj["u"], obj["v"])
    print(f"{obj['name']}: {position}")
```

## 文件说明

### 核心文件
- `vision_based_pick_place.py`: 主要的抓取系统实现
- `vision_pick_example.py`: 使用示例和测试代码
- `simple_pick_place.py`: 原始的抓取基础功能

### 标定文件
- `eye2tcp_matrix.txt`: 手眼标定矩阵 (4x4变换矩阵)
- `intrinsic.txt`: 相机内参矩阵 (3x3内参矩阵)

### 配置参数
- `intrinsic copy.txt`: 相机内参备份文件

## 系统要求

### 硬件要求
- 机械臂（默认IP: 192.168.5.111）
- 夹爪（默认串口: /dev/ttyUSB0）
- 相机（已完成手眼标定）

### 软件依赖
```bash
pip install numpy transforms3d
```

## 配置说明

### 相机内参配置
内参矩阵格式 (intrinsic.txt):
```
fx  0   cx
0   fy  cy  
0   0   1
```

### 手眼标定矩阵配置
4x4变换矩阵格式 (eye2tcp_matrix.txt):
```
R11 R12 R13 Tx
R21 R22 R23 Ty
R31 R32 R33 Tz
0   0   0   1
```

## API参考

### 主要类: VisionBasedPickPlace

#### 初始化
```python
vision_system = VisionBasedPickPlace(robot_ip='192.168.5.111', gripper_port='/dev/ttyUSB0')
```

#### 核心方法
- `calculate_object_position_in_base(distance, u, v)`: 计算物体在基座系下的位置
- `vision_based_pick_and_place(distance, u, v)`: 执行完整的视觉抓取
- `execute_grasp_sequence(target_position)`: 执行抓取序列
- `test_system()`: 系统功能测试

### 简化API
- `create_vision_pick_system()`: 创建系统实例
- `execute_vision_pick()`: 执行抓取
- `calculate_object_position()`: 计算位置

## 使用示例

### 完整示例
```python
#!/usr/bin/env python3
from vision_based_pick_place import VisionBasedPickPlace

# 创建系统
vision_system = VisionBasedPickPlace()

# 系统测试
vision_system.test_system()

# 视觉抓取
distance = 300.0  # mm
pixel_u = 700.0   # 像素坐标
pixel_v = 300.0   # 像素坐标

vision_system.vision_based_pick_and_place(distance, pixel_u, pixel_v)
```

### 运行示例
```bash
# 运行主程序
python vision_based_pick_place.py

# 运行示例程序
python vision_pick_example.py
```

## 抓取流程

### 10步抓取序列
1. **初始化夹爪姿态**: 设置到预定义的抓取姿态
2. **打开夹爪**: 准备抓取
3. **移动到接近位置**: 物体上方安全距离
4. **下降到抓取位置**: 精确到抓取高度
5. **关闭夹爪**: 执行抓取
6. **提升到安全高度**: 避免碰撞
7. **移动到放置位置上方**: 转移到目标区域
8. **下降到放置位置**: 精确放置
9. **打开夹爪**: 释放物体
10. **回到安全位置**: 完成任务

## 安全考虑

### 工作空间限制
- X, Y轴: ±800mm
- Z轴: 50-600mm

### 危险区域避免
- 可配置的禁止区域
- 实时安全检查

### 异常处理
- 运动失败自动回到安全位置
- 详细错误日志输出
- 系统状态监控

## 调试和测试

### 测试功能
```python
# 系统功能测试
vision_system.test_system()

# 批量测试像素坐标转换
test_cases = [
    (300, 637, 379, "图像中心"),
    (300, 700, 300, "右上角"),
    (300, 574, 379, "左侧")
]
vision_system.test_pixel_to_base_conversion(test_cases)
```

### 日志输出
系统提供详细的日志信息：
- 坐标转换过程
- 机械臂运动状态
- 抓取序列执行情况
- 错误和异常信息

## 扩展功能

### 自定义抓取参数
```python
# 修改抓取参数
vision_system.grasp_height_offset = 30  # 抓取高度偏移
vision_system.approach_height = 120     # 接近高度
vision_system.place_position = [400, 200, 150]  # 放置位置
```

### 添加新的抓取策略
可以继承VisionBasedPickPlace类来实现特定的抓取策略。

## 故障排除

### 常见问题
1. **坐标转换异常**: 检查手眼标定矩阵和相机内参
2. **机械臂连接失败**: 确认IP地址和网络连接
3. **夹爪控制失败**: 检查串口连接和权限
4. **位置计算错误**: 验证像素坐标和距离数据

### 解决方案
- 检查标定文件格式
- 验证硬件连接
- 查看详细日志输出
- 使用测试功能验证系统状态

## 版本历史

### v1.0 (2025-07-12)
- 基本的视觉抓取功能
- 手眼标定支持
- 完整的抓取序列
- 详细的日志输出

## 联系方式

如有问题或建议，请联系开发团队。

---

**注意**: 在实际使用前，请确保已完成手眼标定，并验证所有硬件连接正常。
