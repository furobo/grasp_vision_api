#!/bin/bash

echo "=== 快速修复脚本 (Quick Fix Script) ==="
echo "Applying immediate fixes for system issues..."
echo ""

# Quick fix for USB permissions (temporary)
echo "1. 临时修复USB权限 (Temporary USB permission fix)..."
sudo chmod 666 /dev/ttyUSB0
echo "USB permissions temporarily fixed"

# Add user to dialout group for permanent fix
echo "2. 添加用户到dialout组 (Add user to dialout group)..."
sudo usermod -a -G dialout $USER

# Set environment for RealSense
echo "3. 设置环境变量 (Set environment variables)..."
export DISPLAY=localhost:10.0

echo ""
echo "=== 快速修复完成 (Quick fix completed) ==="
echo ""
echo "现在可以尝试运行系统："
echo "Now you can try running the system:"
echo "python calibration_viewer.py"
echo ""
echo "注意：USB权限修复是临时的，重启后需要重新运行"
echo "Note: USB permission fix is temporary, need to re-run after reboot"
echo "要永久修复，请注销并重新登录"
echo "For permanent fix, please logout and login again" 