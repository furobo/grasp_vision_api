#!/usr/bin/env python3
"""
YOLO物体检测和抓取点计算测试脚本
用于测试calibration_viewer.py中新增的YOLO检测功能
"""

import sys
import os
import time

# 添加模块路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def test_yolo_detection():
    """测试YOLO检测功能"""
    print("=" * 60)
    print("YOLO物体检测和抓取点计算测试")
    print("=" * 60)
    
    try:
        # 导入CameraWindow类
        from calibration_viewer import CameraWindow
        print("✓ 成功导入CameraWindow类")
        
        # 创建CameraWindow实例（不显示GUI）
        print("正在初始化相机和系统...")
        camera_window = CameraWindow()
        print("✓ 相机和系统初始化成功")
        
        # 等待系统稳定
        print("等待系统稳定...")
        time.sleep(2)
        
        # 执行YOLO检测
        print("\n开始执行YOLO检测...")
        target_object = "bottle"  # 可以修改为其他物体类别
        
        # 调用检测函数
        success = camera_window.detect_and_grasp_object(target_object)
        
        if success:
            print(f"\n✓ 成功检测到 {target_object} 并计算抓取点!")
            print("检查以下文件获取结果:")
            
            save_dir = "/home/adminpc/Documents/复旦复合机器人/手眼标定demo/robotics_v20250704/robotics/eye_hand_calib/data/yolo_detections"
            
            files_to_check = [
                "rgb.png",           # RGB原始图像
                "depth.npy",         # 深度数据（numpy格式）
                "depth_visualization.png", # 深度可视化图像
                "detection_result.png",    # 检测结果图像
                "point.txt"          # 抓取点坐标和深度
            ]
            
            for filename in files_to_check:
                filepath = os.path.join(save_dir, filename)
                if os.path.exists(filepath):
                    print(f"  ✓ {filepath}")
                else:
                    print(f"  ✗ {filepath} (文件不存在)")
            
            # 读取并显示抓取点信息
            point_file = os.path.join(save_dir, "point.txt")
            if os.path.exists(point_file):
                print(f"\n抓取点信息:")
                with open(point_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) >= 3:
                        x = float(lines[0].strip())
                        y = float(lines[1].strip())
                        depth = float(lines[2].strip())
                        print(f"  中心点坐标: ({x}, {y})")
                        print(f"  深度值: {depth:.2f} mm")
                        print(f"  距离: {depth/1000:.3f} m")
        else:
            print(f"\n✗ 未能检测到 {target_object}")
            print("请检查:")
            print("  1. 相机是否正常连接")
            print("  2. 场景中是否有目标物体")
            print("  3. 光照条件是否良好")
            print("  4. YOLO模型是否正确安装")
        
    except ImportError as e:
        print(f"✗ 导入模块失败: {e}")
        print("请确保:")
        print("  1. 已安装所需的Python包")
        print("  2. 相机驱动正确安装")
        print("  3. 运行: pip install ultralytics")
        
    except Exception as e:
        print(f"✗ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

def test_with_different_objects():
    """测试检测不同类别的物体"""
    print("\n" + "=" * 60)
    print("测试检测不同类别的物体")
    print("=" * 60)
    
    # YOLO支持的常见物体类别
    test_objects = ["bottle", "cup", "bowl", "banana", "apple", "orange", "mouse", "keyboard", "book", "phone"]
    
    try:
        from calibration_viewer import CameraWindow
        camera_window = CameraWindow()
        
        print("可以测试的物体类别:")
        for i, obj in enumerate(test_objects):
            print(f"  {i+1}. {obj}")
        
        while True:
            choice = input(f"\n请选择要检测的物体类别 (1-{len(test_objects)}, 或输入 'q' 退出): ").strip()
            
            if choice.lower() == 'q':
                break
                
            try:
                index = int(choice) - 1
                if 0 <= index < len(test_objects):
                    target_object = test_objects[index]
                    print(f"\n正在检测: {target_object}")
                    success = camera_window.detect_and_grasp_object(target_object)
                    
                    if success:
                        print(f"✓ 成功检测到 {target_object}!")
                    else:
                        print(f"✗ 未检测到 {target_object}")
                else:
                    print("无效的选择，请重新输入")
            except ValueError:
                print("请输入有效的数字")
                
    except Exception as e:
        print(f"错误: {e}")

def main():
    """主函数"""
    print("YOLO物体检测测试程序")
    print("此程序将测试calibration_viewer.py中的YOLO检测功能")
    
    # 检查依赖
    print("\n检查依赖项...")
    
    dependencies = [
        ("ultralytics", "pip install ultralytics"),
        ("cv2", "pip install opencv-python"),
        ("numpy", "pip install numpy"),
        ("yaml", "pip install pyyaml"),
        ("PyQt5", "pip install PyQt5")
    ]
    
    missing_deps = []
    for dep_name, install_cmd in dependencies:
        try:
            if dep_name == "cv2":
                import cv2
            elif dep_name == "ultralytics":
                import ultralytics
            elif dep_name == "numpy":
                import numpy
            elif dep_name == "yaml":
                import yaml
            elif dep_name == "PyQt5":
                from PyQt5.QtWidgets import QApplication
            print(f"✓ {dep_name}")
        except ImportError:
            print(f"✗ {dep_name} - 请运行: {install_cmd}")
            missing_deps.append((dep_name, install_cmd))
    
    if missing_deps:
        print(f"\n发现 {len(missing_deps)} 个缺失的依赖项，请先安装:")
        for dep_name, install_cmd in missing_deps:
            print(f"  {install_cmd}")
        return
    
    print("\n选择测试模式:")
    print("1. 标准测试 (检测bottle)")
    print("2. 自定义物体检测")
    
    choice = input("请选择 (1 或 2): ").strip()
    
    if choice == "1":
        # 运行标准测试
        test_yolo_detection()
    elif choice == "2":
        # 运行自定义物体检测
        test_with_different_objects()
    else:
        print("无效选择，运行标准测试...")
        test_yolo_detection()

if __name__ == "__main__":
    main() 