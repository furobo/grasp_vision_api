sudo chmod 666 /dev/ttyUSB0

#!/bin/bash
CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC2155
export PYTHONPATH=${CURR_DIR}/GripperTestPython:$PYTHONPATH
#!/bin/bash


# common
source /home/adminpc/文档/fudan_robot/复旦复合机器人/手眼标定demo/robotics/hardware/common_lib/env.sh
# 相机
source /home/adminpc/文档/fudan_robot/复旦复合机器人/手眼标定demo/robotics/hardware/camera_orbbec/pyorbbecsdk/env.sh
# 机械臂
source /home/adminpc/文档/fudan_robot/复旦复合机器人/手眼标定demo/robotics/hardware/robot_fr/env.sh
source /home/adminpc/文档/fudan_robot/复旦复合机器人/手眼标定demo/robotics/hardware/robot_duco/env.sh
# grasper 
source /home/adminpc/文档/fudan_robot/复旦复合机器人/手眼标定demo/robotics/hardware/grasper_jodell/env.sh
source /home/adminpc/文档/fudan_robot/复旦复合机器人/手眼标定demo/robotics/hardware/grasper_dh/env.sh

# Project
CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export PYTHONPATH=${CURR_DIR}/hardware/camera_orbbec/lib:$PYTHONPATH
export PYTHONPATH=${CURR_DIR}/hardware/camera_realsenseD435/lib:$PYTHONPATH
export PYTHONPATH=${CURR_DIR}/hardware/robot_fr/lib:$PYTHONPATH

conda activate pick_place
