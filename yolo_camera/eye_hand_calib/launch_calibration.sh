#!/bin/bash
# launch_calibration_x11.sh - 简化的启动脚本

#!/bin/bash

# 设置环境变量
export DISPLAY=localhost:10.0
export QT_X11_NO_MITSHM=1

# 切换到工作目录
cd /home/adminpc/Documents/复旦复合机器人/手眼标定demo/robotics_v20250704/robotics/eye_hand_calib

# 设置Python路径
export PYTHONPATH="/home/adminpc/Documents/复旦复合机器人/手眼标定demo/robotics_v20250704/robotics/eye_hand_calib:$PYTHONPATH"

echo "=== 手眼标定GUI X11转发启动器 ==="
echo "DISPLAY: $DISPLAY"
echo "工作目录: $(pwd)"
echo "Python路径: $PYTHONPATH"
echo ""

# 检查文件是否存在
if [ ! -f "calibration_viewer.py" ]; then
    echo "错误: calibration_viewer.py 文件不存在"
    exit 1
fi



echo "文件检查通过，启动GUI..."
echo "按 Ctrl+C 停止程序"
echo ""

# 启动Python脚本
python3 calibration_viewer.py