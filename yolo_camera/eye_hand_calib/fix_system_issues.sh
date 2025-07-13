#!/bin/bash

echo "=== 复旦复合机器人系统修复脚本 ==="
echo "Fixing system issues for robotic arm grasping system..."
echo ""

# 1. Fix USB serial port permissions
echo "1. 修复USB串口权限 (Fixing USB serial port permissions)..."
sudo usermod -a -G dialout $USER
echo "User added to dialout group. 需要重新登录生效 (Need to re-login for changes to take effect)"

# 2. Fix OpenGL/Mesa driver issues
echo ""
echo "2. 修复OpenGL驱动问题 (Fixing OpenGL driver issues)..."
sudo apt update
sudo apt install -y mesa-utils libgl1-mesa-glx libgl1-mesa-dri mesa-common-dev

# 3. Install RealSense dependencies
echo ""
echo "3. 安装RealSense依赖 (Installing RealSense dependencies)..."
sudo apt install -y librealsense2-utils librealsense2-dev

# 4. Install additional system dependencies
echo ""
echo "4. 安装系统依赖 (Installing system dependencies)..."
sudo apt install -y python3-pyqt5 python3-opencv python3-numpy python3-pip

# 5. Set up Python environment
echo ""
echo "5. 设置Python环境 (Setting up Python environment)..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate camera_yolo

# Install required Python packages
pip install pyrealsense2 opencv-python numpy pyqt5 ultralytics

echo ""
echo "=== 修复完成 (Fix completed) ==="
echo ""
echo "请执行以下步骤完成修复："
echo "Please follow these steps to complete the fix:"
echo ""
echo "1. 重新登录或重启系统以应用权限更改"
echo "   Re-login or restart system to apply permission changes"
echo ""
echo "2. 运行以下命令测试系统："
echo "   Run the following command to test the system:"
echo "   source ~/miniconda3/etc/profile.d/conda.sh"
echo "   conda activate camera_yolo"
echo "   export DISPLAY=localhost:10.0"
echo "   python calibration_viewer.py"
echo ""
echo "如果仍有问题，请检查硬件连接："
echo "If issues persist, please check hardware connections:"
echo "- RealSense camera USB connection"
echo "- DH gripper USB connection (/dev/ttyUSB0)"
echo "- Robot arm network connection (192.168.5.110)" 