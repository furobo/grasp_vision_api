# 手眼标定系统使用说明

## 概述

本系统是一个集成了YOLO物体检测和手眼标定的机械臂抓取系统，支持通过X11转发在本地电脑上显示远程GUI。

## 系统架构

```
本地电脑 (Windows/Linux/Mac)
    ↓ SSH X11转发
远程机械臂主机 (Ubuntu + ROS)
    ├── 手眼标定GUI
    ├── YOLO物体检测
    ├── 深度相机 (RealSense)
    ├── 机械臂控制
    └── 夹爪控制
```

## 文件结构

```
eye_hand_calib/
├── calibration_viewer.py          # 主GUI程序
├── world_image.py                 # 3D坐标转换
├── config.yaml                    # 系统配置
├── launch_gui_x11.sh              # X11转发启动脚本
├── install_camera_yolo_env.sh     # 环境安装脚本
├── requirements_camera_yolo.txt   # 依赖包列表
├── README_使用说明.md            # 本说明文档
└── ViewXXXGroupBox.py            # GUI组件
```

## 快速开始

### 1. 环境准备

#### 1.1 安装conda环境
```bash
# 进入工作目录
cd /home/adminpc/Documents/复旦复合机器人/手眼标定demo/robotics_v20250704/robotics/eye_hand_calib

# 运行安装脚本
chmod +x install_camera_yolo_env.sh
./install_camera_yolo_env.sh
```

#### 1.2 手动安装（可选）
```bash
# 创建conda环境
conda create -n camera_yolo python=3.8 -y

# 激活环境
conda activate camera_yolo

# 安装依赖
pip install -r requirements_camera_yolo.txt
```

### 2. 本地X11服务器设置

#### 2.1 Windows系统
1. 下载并安装 [VcXsrv](https://sourceforge.net/projects/vcxsrv/) 或 [Xming](https://sourceforge.net/projects/xming/)
2. 启动X11服务器，配置如下：
   - Display number: 0
   - Start no client: 选中
   - Clipboard: 选中
   - **Disable access control**: 选中（重要）

#### 2.2 macOS系统
```bash
# 安装XQuartz
brew install --cask xquartz

# 启动XQuartz
open -a XQuartz
```

#### 2.3 Linux系统
```bash
# 通常已经内置X11服务器
echo $DISPLAY
```

### 3. SSH连接设置

```bash
# 从本地电脑连接到机械臂主机
ssh -X adminpc@192.168.5.105

# 或者使用更宽松的X11转发
ssh -Y adminpc@192.168.5.105

# 验证X11转发
echo $DISPLAY
xclock  # 测试窗口显示
```

### 4. 启动GUI

```bash
# 进入工作目录
cd /home/adminpc/Documents/复旦复合机器人/手眼标定demo/robotics_v20250704/robotics/eye_hand_calib

# 运行启动脚本
chmod +x launch_gui_x11.sh
./launch_gui_x11.sh
```

## 详细配置

### 1. 系统配置 (config.yaml)

```yaml
camera:
  type: "realsense"  # 或 "orbbec"

robot:
  type: "DUCO"       # 或 "FAIR"
  ip: "192.168.5.110"
  max_speed: 1.0
  torque: 100

grasper:
  type: "DH"         # 或 "EPG"
  port: "/dev/ttyUSB0"
  max_position: 1000
  min_position: 0
```

### 2. 硬件接口

#### 2.1 支持的相机
- **RealSense D435**: 深度相机，支持RGB-D数据
- **Orbbec**: 深度相机，支持RGB-D数据

#### 2.2 支持的机械臂
- **DUCO机械臂**: 新松机械臂
- **FAIR机械臂**: 法拉第机械臂

#### 2.3 支持的夹爪
- **DH夹爪**: 大寰夹爪
- **EPG夹爪**: 优傲夹爪

## 功能使用

### 1. 手眼标定

1. **拍摄标定图像**
   - 在GUI中点击"拍照"按钮
   - 移动机械臂到不同位置
   - 拍摄多个角度的ArUco标记图像

2. **执行标定**
   - 点击"开始标定"按钮
   - 系统自动计算手眼标定矩阵
   - 结果保存到 `data/eye2tcp_matrix.txt`

### 2. 物体检测

1. **YOLO检测**
   - 系统支持YOLO v8物体检测
   - 可检测常见物体：瓶子、杯子、书本等
   - 输出边界框和置信度

2. **3D位置计算**
   - 结合深度信息计算物体3D位置
   - 坐标转换到机械臂基座标系
   - 输出可直接用于抓取的坐标

### 3. 机械臂控制

1. **关节控制**
   - 支持单关节调整
   - 支持整体姿态调整
   - 实时显示关节角度

2. **TCP控制**
   - 支持笛卡尔空间运动
   - 位置和姿态独立控制
   - 实时显示TCP位置

### 4. 夹爪控制

1. **位置控制**
   - 支持位置调整
   - 速度和力矩设置
   - 实时显示夹爪状态

## 故障排除

### 1. 模块导入错误

```bash
# 检查Python路径
python -c "import sys; print('\n'.join(sys.path))"

# 检查模块是否存在
python -c "import numpy; print('numpy OK')"
python -c "import cv2; print('cv2 OK')"
python -c "import PyQt5; print('PyQt5 OK')"
```

### 2. X11转发问题

```bash
# 检查DISPLAY变量
echo $DISPLAY

# 测试X11连接
xeyes
xclock

# 重新连接SSH
ssh -X adminpc@192.168.5.105
```

### 3. 相机连接问题

```bash
# 检查相机设备
lsusb | grep Intel  # RealSense
lsusb | grep Orbbec  # Orbbec

# 检查相机权限
sudo chmod 666 /dev/video*
```

### 4. 机械臂连接问题

```bash
# 检查网络连接
ping 192.168.5.110

# 检查端口
telnet 192.168.5.110 8080
```

## 开发指南

### 1. 添加新的物体检测模型

```python
# 在 yolo_detector.py 中添加自定义模型
detector = YOLODetector("path/to/your/model.pt")
```

### 2. 添加新的相机接口

```python
# 创建新的相机接口类
class NewCameraInterface:
    def __init__(self):
        # 初始化相机
        pass
    
    def get_camera_intrinsic(self):
        # 返回相机内参
        pass
    
    def read_image_and_depth(self):
        # 返回RGB图像和深度图
        pass
```

### 3. 添加新的机械臂接口

```python
# 创建新的机械臂接口类
class NewRobotInterface:
    def __init__(self, robot_ip):
        # 初始化机械臂连接
        pass
    
    def moveJ(self, joints):
        # 关节运动
        pass
    
    def get_joint_degree(self):
        # 获取关节角度
        pass
```

## 联系支持

如果遇到问题，请检查：
1. 所有依赖包是否正确安装
2. 硬件连接是否正常
3. 网络设置是否正确
4. X11转发是否正常工作

## 更新记录

- **v1.0** (2024-01-XX): 初始版本
- **v1.1** (2024-01-XX): 修复模块导入问题
- **v1.2** (2024-01-XX): 添加X11转发支持
- **v1.3** (2024-01-XX): 集成YOLO检测模块 