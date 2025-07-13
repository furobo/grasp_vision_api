#!/usr/bin/env python3
"""
YOLO检测演示脚本
演示如何使用YOLO检测功能
"""

import sys
import os
import cv2
import numpy as np
import time

# 添加模块路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def demo_yolo_detection():
    """演示YOLO检测功能"""
    print("=" * 60)
    print("YOLO检测功能演示")
    print("=" * 60)
    
    # 1. 导入检测器
    try:
        from grasper_detector import GraspDetector
        print("✓ 成功导入GraspDetector")
    except ImportError as e:
        print(f"✗ 导入GraspDetector失败: {e}")
        return False
    
    # 2. 初始化检测器
    try:
        detector = GraspDetector()
        print("✓ 成功初始化YOLO检测器")
    except Exception as e:
        print(f"✗ 初始化检测器失败: {e}")
        return False
    
    # 3. 测试相机连接
    print("\n正在测试相机连接...")
    try:
        # 尝试连接相机
        if detector.config['camera']['type'] == 'realsense':
            from RealSenceInterface import RealSenseInterface
            camera = RealSenseInterface()
        elif detector.config['camera']['type'] == 'orbbec':
            from orbbec_sdk_interface import OrbbecSDKInterface
            camera = OrbbecSDKInterface()
        else:
            print("使用OpenCV相机")
            camera = cv2.VideoCapture(0)
        
        print("✓ 相机连接成功")
    except Exception as e:
        print(f"✗ 相机连接失败: {e}")
        print("使用模拟图像进行演示...")
        camera = None
    
    # 4. 演示检测功能
    print("\n开始检测演示...")
    
    if camera is not None:
        # 使用真实相机
        print("使用真实相机画面进行检测...")
        
        if hasattr(camera, 'read_image_and_depth'):
            # RealSense或Orbbec相机
            for i in range(5):
                print(f"\n第 {i+1} 次检测:")
                
                try:
                    # 读取图像
                    rgb_image, depth_image = camera.read_image_and_depth()
                    
                    # 运行检测
                    start_time = time.time()
                    candidates = detector.detect_objects(rgb_image, depth_image)
                    end_time = time.time()
                    
                    # 显示结果
                    print(f"  检测耗时: {(end_time - start_time)*1000:.1f}ms")
                    print(f"  检测到 {len(candidates)} 个目标")
                    
                    for j, candidate in enumerate(candidates):
                        print(f"    目标 {j+1}: {candidate['class_name']} "
                              f"(置信度: {candidate['confidence']:.2f})")
                        if candidate['position_3d']:
                            x, y, z = candidate['position_3d']
                            print(f"      3D位置: ({x:.2f}, {y:.2f}, {z:.2f})")
                    
                    # 保存检测结果
                    if candidates:
                        vis_image = detector.visualize_detections(rgb_image, candidates)
                        cv2.imwrite(f"demo_result_{i+1}.jpg", cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
                        print(f"  结果已保存到 demo_result_{i+1}.jpg")
                    
                    time.sleep(1)  # 等待1秒
                    
                except Exception as e:
                    print(f"  检测失败: {e}")
        
        else:
            # OpenCV相机
            print("使用OpenCV相机...")
            for i in range(5):
                print(f"\n第 {i+1} 次检测:")
                
                try:
                    ret, frame = camera.read()
                    if not ret:
                        print("  无法读取相机画面")
                        continue
                    
                    # 转换颜色空间
                    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    depth_image = np.ones((rgb_image.shape[0], rgb_image.shape[1])) * 1000
                    
                    # 运行检测
                    start_time = time.time()
                    candidates = detector.detect_objects(rgb_image, depth_image)
                    end_time = time.time()
                    
                    # 显示结果
                    print(f"  检测耗时: {(end_time - start_time)*1000:.1f}ms")
                    print(f"  检测到 {len(candidates)} 个目标")
                    
                    for j, candidate in enumerate(candidates):
                        print(f"    目标 {j+1}: {candidate['class_name']} "
                              f"(置信度: {candidate['confidence']:.2f})")
                    
                    # 保存检测结果
                    if candidates:
                        vis_image = detector.visualize_detections(rgb_image, candidates)
                        cv2.imwrite(f"demo_result_{i+1}.jpg", cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
                        print(f"  结果已保存到 demo_result_{i+1}.jpg")
                    
                    time.sleep(1)  # 等待1秒
                    
                except Exception as e:
                    print(f"  检测失败: {e}")
    
    else:
        # 使用模拟图像
        print("使用模拟图像进行检测...")
        
        # 创建一个测试图像
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 在图像上绘制一些形状
        cv2.rectangle(test_image, (100, 100), (300, 300), (0, 255, 0), 2)
        cv2.circle(test_image, (400, 200), 50, (0, 0, 255), -1)
        cv2.putText(test_image, "Demo Image", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 创建模拟深度图
        depth_image = np.ones((480, 640)) * 1000
        
        # 运行检测
        print("\n运行检测...")
        start_time = time.time()
        candidates = detector.detect_objects(test_image, depth_image)
        end_time = time.time()
        
        # 显示结果
        print(f"检测耗时: {(end_time - start_time)*1000:.1f}ms")
        print(f"检测到 {len(candidates)} 个目标")
        
        for j, candidate in enumerate(candidates):
            print(f"  目标 {j+1}: {candidate['class_name']} "
                  f"(置信度: {candidate['confidence']:.2f})")
            if candidate['position_3d']:
                x, y, z = candidate['position_3d']
                print(f"    3D位置: ({x:.2f}, {y:.2f}, {z:.2f})")
        
        # 保存检测结果
        if candidates:
            vis_image = detector.visualize_detections(test_image, candidates)
            cv2.imwrite("demo_result.jpg", cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            print("结果已保存到 demo_result.jpg")
        
        cv2.imwrite("demo_original.jpg", cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR))
        print("原始图像已保存到 demo_original.jpg")
    
    # 5. 性能测试
    print("\n" + "=" * 60)
    print("性能测试")
    print("=" * 60)
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    depth_image = np.ones((480, 640)) * 1000
    
    # 运行多次检测测试性能
    times = []
    for i in range(10):
        start_time = time.time()
        candidates = detector.detect_objects(test_image, depth_image)
        end_time = time.time()
        times.append((end_time - start_time) * 1000)
    
    print(f"10次检测平均耗时: {np.mean(times):.1f}ms")
    print(f"最快检测时间: {np.min(times):.1f}ms")
    print(f"最慢检测时间: {np.max(times):.1f}ms")
    print(f"检测频率: {1000/np.mean(times):.1f} FPS")
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)
    
    return True

def main():
    """主函数"""
    print("YOLO检测演示程序")
    print("此程序将演示YOLO检测功能的使用")
    
    # 检查依赖
    print("\n检查依赖项...")
    try:
        import ultralytics
        print("✓ ultralytics")
    except ImportError:
        print("✗ ultralytics - 请运行: pip install ultralytics")
        return
    
    try:
        import yaml
        print("✓ yaml")
    except ImportError:
        print("✗ yaml - 请运行: pip install pyyaml")
        return
    
    try:
        import cv2
        print("✓ opencv-python")
    except ImportError:
        print("✗ opencv-python - 请运行: pip install opencv-python")
        return
    
    try:
        import numpy
        print("✓ numpy")
    except ImportError:
        print("✗ numpy - 请运行: pip install numpy")
        return
    
    # 运行演示
    input("\n按回车键开始演示...")
    demo_yolo_detection()

if __name__ == "__main__":
    main() 