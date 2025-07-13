#!/bin/bash
# install_camera_yolo_env.sh - 安装camera_yolo环境

echo "=== 手眼标定系统环境安装 ==="
echo "时间: $(date)"
echo ""

# 1. 检查conda
echo "1. 检查conda..."
if ! command -v conda >/dev/null 2>&1; then
    echo "   ✗ conda未找到，请先安装Anaconda或Miniconda"
    exit 1
else
    echo "   ✓ conda可用: $(conda --version)"
fi

# 2. 检查camera_yolo环境
echo ""
echo "2. 检查camera_yolo环境..."
if conda info --envs | grep -q "camera_yolo"; then
    echo "   ✓ camera_yolo环境已存在"
    read -p "   是否重新创建环境？(y/n): " recreate
    if [[ $recreate =~ ^[Yy]$ ]]; then
        echo "   删除现有环境..."
        conda env remove -n camera_yolo -y
        echo "   ✓ 环境已删除"
    else
        echo "   跳过环境创建"
        skip_create=true
    fi
else
    echo "   - camera_yolo环境不存在"
fi

# 3. 创建环境
if [ "$skip_create" != true ]; then
    echo ""
    echo "3. 创建camera_yolo环境..."
    conda create -n camera_yolo python=3.8 -y
    if [ $? -eq 0 ]; then
        echo "   ✓ 环境创建成功"
    else
        echo "   ✗ 环境创建失败"
        exit 1
    fi
fi

# 4. 激活环境
echo ""
echo "4. 激活环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate camera_yolo

if [ "$CONDA_DEFAULT_ENV" = "camera_yolo" ]; then
    echo "   ✓ 环境激活成功: $CONDA_DEFAULT_ENV"
else
    echo "   ✗ 环境激活失败"
    exit 1
fi

# 5. 更新pip
echo ""
echo "5. 更新pip..."
python -m pip install --upgrade pip
echo "   ✓ pip更新完成"

# 6. 安装依赖包
echo ""
echo "6. 安装依赖包..."
echo "   这可能需要几分钟时间..."

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements_camera_yolo.txt"

if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "   使用requirements文件: $REQUIREMENTS_FILE"
    pip install -r "$REQUIREMENTS_FILE"
else
    echo "   requirements文件不存在，手动安装核心包..."
    pip install numpy opencv-python PyQt5 PyYAML transforms3d open3d pyvista pyvistaqt pyrealsense2 OpenEXR matplotlib tqdm scipy pillow
fi

# 7. 检查安装
echo ""
echo "7. 检查安装..."
modules=("numpy" "cv2" "PyQt5" "yaml" "transforms3d" "open3d" "pyvista" "pyrealsense2" "OpenEXR")
failed_modules=()

for module in "${modules[@]}"; do
    if python -c "import $module" 2>/dev/null; then
        echo "   ✓ $module"
    else
        echo "   ✗ $module (失败)"
        failed_modules+=("$module")
    fi
done

# 8. 检查可选模块
echo ""
echo "8. 检查可选模块..."
optional_modules=("ultralytics" "torch" "torchvision")
for module in "${optional_modules[@]}"; do
    if python -c "import $module" 2>/dev/null; then
        echo "   ✓ $module (可选)"
    else
        echo "   - $module (未安装，可选)"
    fi
done

# 9. 完成
echo ""
echo "9. 安装完成！"
if [ ${#failed_modules[@]} -eq 0 ]; then
    echo "   ✓ 所有核心模块安装成功"
    echo ""
    echo "使用方法:"
    echo "   conda activate camera_yolo"
    echo "   cd /path/to/eye_hand_calib"
    echo "   ./launch_gui_x11.sh"
else
    echo "   ✗ 以下模块安装失败："
    for module in "${failed_modules[@]}"; do
        echo "     - $module"
    done
    echo ""
    echo "请手动安装失败的模块："
    echo "   conda activate camera_yolo"
    echo "   pip install <module_name>"
fi

echo ""
echo "完成时间: $(date)" 