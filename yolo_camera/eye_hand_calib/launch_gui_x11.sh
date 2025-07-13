#!/bin/bash
# launch_gui_x11.sh - X11转发启动脚本
# 用于在本地电脑上显示远程的手眼标定GUI

echo "=== 手眼标定GUI X11转发启动器 ==="
echo "时间: $(date)"
echo ""

# 1. 检查SSH X11转发
echo "1. 检查SSH X11转发..."
if [ -z "$SSH_CLIENT" ]; then
    echo "   警告: 未检测到SSH连接"
else
    echo "   ✓ SSH连接: $SSH_CLIENT"
fi

if [ -z "$DISPLAY" ]; then
    echo "   ✗ DISPLAY变量未设置"
    echo "   请确保SSH连接启用了X11转发:"
    echo "   ssh -X adminpc@192.168.5.105"
    echo "   或者 ssh -Y adminpc@192.168.5.105"
    exit 1
else
    echo "   ✓ DISPLAY: $DISPLAY"
fi

# 2. 测试X11连接
echo ""
echo "2. 测试X11连接..."
if command -v xdpyinfo >/dev/null 2>&1; then
    if xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
        echo "   ✓ X11连接测试成功"
    else
        echo "   ✗ X11连接测试失败"
        echo "   请检查X11服务器是否在本地运行"
        exit 1
    fi
else
    echo "   - xdpyinfo命令不可用，跳过测试"
fi

# 3. 检查并激活conda环境
echo ""
echo "3. 检查conda环境..."
if command -v conda >/dev/null 2>&1; then
    echo "   ✓ conda可用"
    
    # 检查camera_yolo环境是否存在
    if conda info --envs | grep -q "camera_yolo"; then
        echo "   ✓ camera_yolo环境存在"
        
        # 激活环境
        echo "   激活camera_yolo环境..."
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate camera_yolo
        
        if [ "$CONDA_DEFAULT_ENV" = "camera_yolo" ]; then
            echo "   ✓ 环境激活成功: $CONDA_DEFAULT_ENV"
        else
            echo "   ✗ 环境激活失败"
            exit 1
        fi
    else
        echo "   ✗ camera_yolo环境不存在"
        echo "   请创建环境: conda create -n camera_yolo python=3.8"
        exit 1
    fi
else
    echo "   - conda不可用，使用系统Python"
fi

# 4. 设置环境变量
echo ""
echo "4. 设置环境变量..."
export QT_X11_NO_MITSHM=1
export QT_DEBUG_PLUGINS=0
export XDG_RUNTIME_DIR="/tmp/runtime-$(id -u)"
mkdir -p "$XDG_RUNTIME_DIR"

echo "   ✓ QT_X11_NO_MITSHM: $QT_X11_NO_MITSHM"
echo "   ✓ XDG_RUNTIME_DIR: $XDG_RUNTIME_DIR"

# 5. 切换到工作目录
WORK_DIR="/home/adminpc/Documents/复旦复合机器人/手眼标定demo/robotics_v20250704/robotics/eye_hand_calib"
echo ""
echo "5. 切换到工作目录..."
if [ -d "$WORK_DIR" ]; then
    cd "$WORK_DIR"
    echo "   ✓ 工作目录: $(pwd)"
else
    echo "   ✗ 工作目录不存在: $WORK_DIR"
    exit 1
fi

# 6. 检查必要文件
echo ""
echo "6. 检查必要文件..."
required_files=("calibration_viewer.py" "config.yaml" "world_image.py")
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✓ $file"
    else
        echo "   ✗ $file (缺失)"
        exit 1
    fi
done

# 7. 检查硬件模块
echo ""
echo "7. 检查硬件模块..."
hardware_paths=(
    "../hardware/common_lib/camera_util.py"
    "../hardware/robot_duco/SiasunRobot.py"
    "../hardware/robot_fr/fairino-python-sdk-master/linux/fairino"
)

for path in "${hardware_paths[@]}"; do
    if [ -e "$path" ]; then
        echo "   ✓ $(basename "$path")"
    else
        echo "   ✗ $(basename "$path") (缺失)"
    fi
done

# 8. 设置Python路径
echo ""
echo "8. 设置Python路径..."
export PYTHONPATH="$WORK_DIR:$PYTHONPATH"
echo "   ✓ PYTHONPATH已设置"

# 9. 检查Python模块
echo ""
echo "9. 检查Python模块..."
modules=("numpy" "cv2" "PyQt5" "yaml" "transforms3d")
for module in "${modules[@]}"; do
    if python -c "import $module" 2>/dev/null; then
        echo "   ✓ $module"
    else
        echo "   ✗ $module (缺失)"
    fi
done

# 10. 启动GUI
echo ""
echo "10. 启动GUI..."
echo "================================================"
echo "GUI启动中..."
echo "GUI窗口应该会在您的本地电脑上显示"
echo "如果窗口没有出现，请检查X11服务器设置"
echo "按 Ctrl+C 停止程序"
echo "================================================"
echo ""

# 运行程序
python calibration_viewer.py

echo ""
echo "程序已退出" 