#!/usr/bin/env python3
"""
简化的物体检测测试脚本
用于快速测试物体检测和抓取点计算功能
"""

import os
import sys
import cv2
import numpy as np

# 添加模块路径
current_dir = os.path.dirname(os.path.abspath(__file__))
hardware_dir = os.path.join(current_dir, "..", "hardware")
camera_realsense_dir = os.path.join(hardware_dir, "camera_realsenseD435")
camera_realsense_lib_dir = os.path.join(hardware_dir, "camera_realsenseD435", "lib")
camera_util_dir = os.path.join(hardware_dir, "common_lib")

# 添加到sys.path
for path in [camera_realsense_dir, camera_realsense_lib_dir, camera_util_dir]:
    abs_path = os.path.abspath(path)
    if abs_path not in sys.path:
        sys.path.append(abs_path)

def test_camera():
    """测试相机功能"""
    print("=" * 50)
    print("测试相机功能")
    print("=" * 50)
    
    try:
        from RealSenceInterface import RealSenseInterface
        camera = RealSenseInterface()
        print("✓ 相机初始化成功")
        
        # 读取一帧图像
        rgb_image, depth_image = camera.read_image_and_depth()
        print(f"✓ 图像读取成功 - RGB: {rgb_image.shape}, 深度: {depth_image.shape}")
        
        # 保存测试图像
        save_dir = "/home/adminpc/Documents/复旦复合机器人/手眼标定demo/robotics_v20250704/robotics/eye_hand_calib/data/yolo_detections"
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存RGB图像
        rgb_path = os.path.join(save_dir, 'test_rgb.png')
        cv2.imwrite(rgb_path, rgb_image)
        print(f"✓ RGB图像已保存到: {rgb_path}")
        
        # 保存深度图像
        depth_path = os.path.join(save_dir, 'test_depth.npy')
        np.save(depth_path, depth_image)
        print(f"✓ 深度图像已保存到: {depth_path}")
        
        return rgb_image, depth_image
        
    except Exception as e:
        print(f"✗ 相机测试失败: {e}")
        return None, None

def test_yolo_detection(rgb_image):
    """测试YOLO检测功能"""
    print("\n" + "=" * 50)
    print("测试YOLO检测功能")
    print("=" * 50)
    
    try:
        from ultralytics import YOLO
        
        # 加载模型
        model_path = os.path.join(current_dir, "yolov8n.pt")
        if os.path.exists(model_path):
            model = YOLO(model_path)
        else:
            model = YOLO('yolov8n.pt')
        print("✓ YOLO模型加载成功")
        
        # 运行检测
        results = model(rgb_image, verbose=False)
        print("✓ YOLO检测完成")
        
        # 处理结果
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    confidence = boxes.conf[i].cpu().numpy()
                    class_id = int(boxes.cls[i].cpu().numpy())
                    class_name = model.names[class_id].lower()
                    
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence),
                        'class_name': class_name,
                        'class_id': class_id
                    }
                    detections.append(detection)
                    print(f"  检测到 {class_name}: 置信度={confidence:.3f}")
        
        print(f"✓ 共检测到 {len(detections)} 个物体")
        return detections
        
    except Exception as e:
        print(f"✗ YOLO检测失败: {e}")
        return []

def test_center_point_calculation(detections, depth_image):
    """测试中心点计算功能"""
    print("\n" + "=" * 50)
    print("测试中心点计算功能")
    print("=" * 50)
    
    if not detections:
        print("✗ 没有检测结果，无法计算中心点")
        return None
    
    # 选择置信度最高的检测结果
    best_detection = max(detections, key=lambda x: x['confidence'])
    print(f"✓ 选择置信度最高的物体: {best_detection['confidence']:.3f}")
    
    # 计算中心点
    bbox = best_detection['bbox']
    center_x = int((bbox[0] + bbox[2]) / 2)
    center_y = int((bbox[1] + bbox[3]) / 2)
    
    # 获取深度值
    if (center_x >= 0 and center_x < depth_image.shape[1] and 
        center_y >= 0 and center_y < depth_image.shape[0]):
        depth = depth_image[center_y, center_x]
        print(f"✓ 中心点: ({center_x}, {center_y}), 深度: {depth:.2f}mm")
        
        # 保存到文件
        point_file = "/home/adminpc/Documents/复旦复合机器人/手眼标定demo/robotics_v20250704/robotics/eye_hand_calib/data/point.txt"
        with open(point_file, 'w') as f:
            f.write(f"{center_x}\n")
            f.write(f"{center_y}\n")
            f.write(f"{depth}\n")
        print(f"✓ 抓取点信息已保存到: {point_file}")
        
        return center_x, center_y, depth
    else:
        print(f"✗ 中心点坐标超出图像范围: ({center_x}, {center_y})")
        return None

def main():
    """主函数"""
    print("物体检测功能测试程序")
    print("此程序将测试相机、YOLO检测和中心点计算功能")
    
    # 1. 测试相机
    rgb_image, depth_image = test_camera()
    if rgb_image is None:
        print("✗ 相机测试失败，程序退出")
        return
    
    # 2. 测试YOLO检测
    detections = test_yolo_detection(rgb_image)
    
    # 3. 测试中心点计算
    center_point = test_center_point_calculation(detections, depth_image)
    
    print("\n" + "=" * 50)
    if center_point:
        print("✓ 所有测试通过!")
    else:
        print("✗ 部分测试失败!")
    print("=" * 50)

if __name__ == "__main__":
    main() 