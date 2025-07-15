#!/bin/bash
# GroundingDINO环境安装脚本

echo "======================================"
echo "GroundingDINO环境安装脚本"
echo "======================================"

# 检查Python版本
echo "检查Python版本..."
python_version=$(python3 --version 2>&1)
echo "当前Python版本: $python_version"

# 检查是否有GPU
echo "检查CUDA环境..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ 检测到NVIDIA GPU"
    nvidia-smi
else
    echo "⚠️  未检测到NVIDIA GPU，将使用CPU模式"
fi

# 创建虚拟环境（可选）
read -p "是否创建新的虚拟环境? (y/N): " create_env
if [[ $create_env =~ ^[Yy]$ ]]; then
    echo "创建虚拟环境..."
    python3 -m venv grounding_dino_env
    source grounding_dino_env/bin/activate
    echo "✓ 虚拟环境已创建并激活"
fi

# 升级pip
echo "升级pip..."
pip install --upgrade pip

# 安装PyTorch (根据CUDA版本选择)
echo "安装PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    # 检查CUDA版本
    cuda_version=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
    echo "检测到CUDA版本: $cuda_version"
    
    if [[ $(echo "$cuda_version >= 11.8" | bc -l) -eq 1 ]]; then
        echo "安装CUDA 11.8版本的PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        echo "安装CPU版本的PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
else
    echo "安装CPU版本的PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# 安装transformers（推荐方式）
echo "安装transformers..."
pip install transformers

# 安装其他必要依赖
echo "安装其他依赖..."
pip install opencv-python pillow numpy pyyaml

# 可选：安装原生GroundingDINO
read -p "是否安装原生GroundingDINO库? (y/N): " install_native
if [[ $install_native =~ ^[Yy]$ ]]; then
    echo "克隆GroundingDINO仓库..."
    git clone https://github.com/IDEA-Research/GroundingDINO.git
    cd GroundingDINO
    
    echo "安装GroundingDINO..."
    pip install -e .
    
    echo "下载预训练模型..."
    wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
    
    echo "下载配置文件..."
    cp groundingdino/config/GroundingDINO_SwinT_OGC.py ../
    cp groundingdino_swint_ogc.pth ../
    
    cd ..
    echo "✓ 原生GroundingDINO安装完成"
fi

# 测试安装
echo "测试安装..."
python3 -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')

try:
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    print('✓ transformers安装成功')
except ImportError as e:
    print(f'✗ transformers安装失败: {e}')

try:
    import cv2
    print('✓ opencv-python安装成功')
except ImportError as e:
    print(f'✗ opencv-python安装失败: {e}')

try:
    from PIL import Image
    print('✓ pillow安装成功')
except ImportError as e:
    print(f'✗ pillow安装失败: {e}')

try:
    import numpy as np
    print('✓ numpy安装成功')
except ImportError as e:
    print(f'✗ numpy安装失败: {e}')
"

echo "======================================"
echo "安装完成！"
echo "======================================"
echo "现在可以运行GroundingDINO检测脚本了："
echo "python3 grounding_dino_detection_and_grasping.py"
echo ""
echo "注意事项："
echo "1. 首次运行时会自动下载模型，需要网络连接"
echo "2. 模型文件较大，请耐心等待"
echo "3. 建议使用GPU以获得更好的性能"
