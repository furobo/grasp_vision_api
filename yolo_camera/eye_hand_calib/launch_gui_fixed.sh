#!/bin/bash
# 修复Qt问题的启动脚本

echo "启动机器人标定GUI（修复版）"

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate camera_yolo

# 设置显示
export DISPLAY=localhost:10.0

# 设置Qt环境变量
export QT_QPA_PLATFORM=xcb
export QT_XCB_GL_INTEGRATION=none
export QT_LOGGING_RULES='qt5ct.debug=false'
export QT_AUTO_SCREEN_SCALE_FACTOR=0
export QT_SCALE_FACTOR=1
export QT_SCREEN_SCALE_FACTORS=""

# 禁用OpenCV的Qt后端
export OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0

# 设置USB权限
sudo chmod 666 /dev/ttyUSB* 2>/dev/null || echo "注意: 未找到USB设备"

# 进入工作目录
cd "$(dirname "$0")"

echo "环境变量设置完成，启动GUI..."
python calibration_viewer.py

