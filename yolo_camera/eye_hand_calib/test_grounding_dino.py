#!/usr/bin/env python3
"""
GroundingDINO安装和功能测试脚本
"""

import sys
import os
import numpy as np
from PIL import Image
import torch

def test_basic_imports():
    """测试基本库导入"""
    print("=" * 50)
    print("测试基本库导入")
    print("=" * 50)
    
    try:
        import torch
        print(f"✓ PyTorch版本: {torch.__version__}")
        print(f"✓ CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA版本: {torch.version.cuda}")
            print(f"✓ GPU数量: {torch.cuda.device_count()}")
    except ImportError as e:
        print(f"✗ PyTorch导入失败: {e}")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV版本: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV导入失败: {e}")
        return False
    
    try:
        from PIL import Image
        print(f"✓ Pillow已安装")
    except ImportError as e:
        print(f"✗ Pillow导入失败: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy版本: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy导入失败: {e}")
        return False
    
    return True

def test_transformers_grounding_dino():
    """测试transformers版本的GroundingDINO"""
    print("\n" + "=" * 50)
    print("测试transformers版本GroundingDINO")
    print("=" * 50)
    
    try:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        print("✓ transformers库导入成功")
        
        # 测试模型加载
        model_name = "IDEA-Research/grounding-dino-base"
        print(f"正在加载模型: {model_name}")
        
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
        
        print("✓ 模型加载成功")
        print(f"✓ 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        return True, processor, model
        
    except ImportError as e:
        print(f"✗ transformers导入失败: {e}")
        return False, None, None
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return False, None, None

def test_detection_with_sample_image():
    """使用示例图像测试检测功能"""
    print("\n" + "=" * 50)
    print("测试物体检测功能")
    print("=" * 50)
    
    # 测试transformers版本
    success, processor, model = test_transformers_grounding_dino()
    if not success:
        return False
    
    try:
        # 创建一个测试图像
        test_image = Image.new('RGB', (640, 480), color='white')
        
        # 绘制一个简单的矩形作为"bottle"
        import cv2
        import numpy as np
        
        img_array = np.array(test_image)
        cv2.rectangle(img_array, (200, 150), (400, 350), (0, 0, 255), -1)  # 红色矩形
        test_image = Image.fromarray(img_array)
        
        print("✓ 测试图像创建成功")
        
        # 进行检测
        text_prompt = "bottle"
        print(f"正在检测: '{text_prompt}'")
        
        inputs = processor(images=test_image, text=text_prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 后处理
        target_sizes = torch.tensor([test_image.size[::-1]])
        results = processor.post_process_grounded_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.3
        )
        
        print(f"✓ 检测完成，检测到 {len(results[0]['boxes'])} 个物体")
        
        for i, (box, score, label) in enumerate(zip(results[0]['boxes'], results[0]['scores'], results[0]['labels'])):
            print(f"  物体 {i+1}: {label}, 置信度: {score:.3f}, 位置: {box}")
        
        return True
        
    except Exception as e:
        print(f"✗ 检测测试失败: {e}")
        return False

def test_camera_interface():
    """测试相机接口"""
    print("\n" + "=" * 50)
    print("测试相机接口")
    print("=" * 50)
    
    try:
        # 尝试导入相机接口
        current_dir = os.path.dirname(os.path.abspath(__file__))
        hardware_dir = os.path.join(current_dir, "..", "hardware")
        
        # 添加各个硬件模块路径
        camera_realsense_dir = os.path.join(hardware_dir, "camera_realsenseD435")
        camera_realsense_lib_dir = os.path.join(hardware_dir, "camera_realsenseD435", "lib")
        camera_util_dir = os.path.join(hardware_dir, "common_lib")
        
        # 添加到sys.path
        for path in [camera_realsense_dir, camera_realsense_lib_dir, camera_util_dir]:
            abs_path = os.path.abspath(path)
            if abs_path not in sys.path:
                sys.path.append(abs_path)
        
        from RealSenceInterface import RealSenseInterface
        print("✓ 相机接口导入成功")
        
        # 不实际初始化相机，只测试导入
        print("✓ 相机接口可用（未实际连接）")
        return True
        
    except ImportError as e:
        print(f"⚠️  相机接口导入失败: {e}")
        print("   这通常是因为相机硬件未连接或RealSense SDK未安装")
        return False
    except Exception as e:
        print(f"⚠️  相机接口测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("GroundingDINO安装和功能测试")
    print("=" * 50)
    
    # 记录测试结果
    results = []
    
    # 测试基本库
    results.append(("基本库导入", test_basic_imports()))
    
    # 测试GroundingDINO
    results.append(("GroundingDINO模型", test_detection_with_sample_image()))
    
    # 测试相机接口
    results.append(("相机接口", test_camera_interface()))
    
    # 显示测试结果
    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{test_name:20s}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 所有测试通过！可以正常使用GroundingDINO检测功能")
    else:
        print("⚠️  部分测试失败，请检查安装是否完整")
    print("=" * 50)
    
    return all_passed

if __name__ == "__main__":
    main()
