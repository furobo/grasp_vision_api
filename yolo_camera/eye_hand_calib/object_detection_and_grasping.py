#!/usr/bin/env python3
"""
物体检测和抓取点计算脚本
功能：
1. 读取深度相机画面中的物体
2. 保存第一帧的RGB图像和深度图像
3. 使用YOLO模型识别指定物体
4. 计算物体中心点坐标和深度
5. 将结果保存到指定文件
"""

import os
import sys
import cv2
import numpy as np
import time
import yaml
from pathlib import Path

# 添加模块路径
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

# 导入相机接口
from RealSenceInterface import RealSenseInterface

class ObjectDetectionAndGrasping:
    """物体检测和抓取点计算类"""
    
    def __init__(self, target_class="cellphone"):
        """
        初始化检测器
        
        Args:
            target_class (str): 目标物体类别，默认为"bottle"
        """
        self.target_class = target_class
        self.save_dir = "/home/adminpc/Documents/复旦复合机器人/手眼标定demo/robotics_v20250704/robotics/eye_hand_calib/data/yolo_detections"
        self.point_file = "/home/adminpc/Documents/复旦复合机器人/手眼标定demo/robotics_v20250704/robotics/eye_hand_calib/data/point.txt"
        
        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 初始化相机
        self.camera = None
        self.init_camera()
        
        # 初始化YOLO模型
        self.yolo_model = None
        self.init_yolo_model()
    
    def init_camera(self):
        """初始化深度相机"""
        try:
            print("正在初始化RealSense相机...")
            self.camera = RealSenseInterface()
            print("✓ 相机初始化成功")
        except Exception as e:
            print(f"✗ 相机初始化失败: {e}")
            self.camera = None
    
    def init_yolo_model(self):
        """初始化YOLO模型"""
        try:
            from ultralytics import YOLO
            # 加载预训练的YOLOv8模型
            model_path = os.path.join(current_dir, "yolov8n.pt")
            if os.path.exists(model_path):
                self.yolo_model = YOLO(model_path)
            else:
                # 如果本地没有模型文件，下载预训练模型
                self.yolo_model = YOLO('yolov8n.pt')
            print("✓ YOLO模型初始化成功")
        except ImportError:
            print("✗ 未安装ultralytics库，请运行: pip install ultralytics")
            self.yolo_model = None
        except Exception as e:
            print(f"✗ YOLO模型初始化失败: {e}")
            self.yolo_model = None
    
    def capture_and_save_images(self):
        """
        捕获并保存RGB图像和深度图像（改进版：多帧平均+滤波）
        
        Returns:
            tuple: (rgb_image, depth_image) 或 (None, None) 如果失败
        """
        if self.camera is None:
            print("✗ 相机未初始化")
            return None, None
        
        try:
            print("正在捕获图像（多帧平均以提高稳定性）...")
            
            # 多帧平均获取稳定的深度图像
            rgb_image, depth_image = self.capture_stable_depth_image()
            
            if rgb_image is None or depth_image is None:
                print("✗ 无法读取图像数据")
                return None, None
            
            print(f"✓ 图像捕获成功 - RGB: {rgb_image.shape}, 深度: {depth_image.shape}")
            print(f"✓ 深度范围: {depth_image.min():.2f} - {depth_image.max():.2f} mm")
            
            # 保存RGB图像
            rgb_path = os.path.join(self.save_dir, 'rgb.png')
            # 注意：RealSense返回的是BGR格式，需要转换为RGB再保存
            rgb_to_save = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(rgb_path, cv2.cvtColor(rgb_to_save, cv2.COLOR_RGB2BGR))
            print(f"✓ RGB图像已保存到: {rgb_path}")
            
            # 保存深度图像（以.npy格式保存原始数据）
            depth_npy_path = os.path.join(self.save_dir, 'depth.npy')
            np.save(depth_npy_path, depth_image)
            print(f"✓ 深度数据已保存到: {depth_npy_path}")
            
            # 保存深度可视化图像
            depth_vis_path = os.path.join(self.save_dir, 'depth_visualization.png')
            depth_vis = np.clip(depth_image, 0, 5000)  # 限制到5米
            depth_vis = (255 * depth_vis / 5000).astype(np.uint8)
            depth_vis_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            cv2.imwrite(depth_vis_path, depth_vis_color)
            print(f"✓ 深度可视化图像已保存到: {depth_vis_path}")
            
            return rgb_to_save, depth_image
            
        except Exception as e:
            print(f"✗ 图像捕获失败: {e}")
            return None, None
    
    def capture_stable_depth_image(self, num_frames=20, skip_frames=50):
        """
        多帧平均获取稳定的深度图像
        
        Args:
            num_frames (int): 用于平均的帧数
            skip_frames (int): 跳过的初始帧数（等待相机稳定）
            
        Returns:
            tuple: (rgb_image, depth_image) 或 (None, None) 如果失败
        """
        try:
            # 跳过前几帧，等待相机稳定
            print(f"  跳过前 {skip_frames} 帧等待相机稳定...")
            for _ in range(skip_frames):
                self.camera.pipeline.wait_for_frames()
            
            # 收集多帧深度图像
            depth_frames = []
            rgb_image = None
            
            print(f"  收集 {num_frames} 帧进行平均...")
            for i in range(num_frames):
                # 读取单帧
                frames = self.camera.pipeline.wait_for_frames()
                aligned_frames = self.camera.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    print(f"    第 {i+1} 帧读取失败，跳过")
                    continue
                
                # 获取RGB图像（只取第一帧）
                if rgb_image is None:
                    rgb_image = np.asanyarray(color_frame.get_data())
                
                # 获取深度图像
                depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32)
                
                # 应用深度尺度
                depth_image = depth_image * self.camera.depth_scale
                depth_image = np.clip(depth_image, 0, self.camera.max_depth)
                
                # 检查深度图像质量
                valid_pixels = np.count_nonzero(depth_image > 0)
                total_pixels = depth_image.size
                valid_ratio = valid_pixels / total_pixels
                
                if valid_ratio > 0.1:  # 至少10%的像素有效
                    depth_frames.append(depth_image)
                    print(f"    第 {i+1} 帧有效像素比例: {valid_ratio:.2%}")
                else:
                    print(f"    第 {i+1} 帧有效像素比例过低: {valid_ratio:.2%}，跳过")
            
            if not depth_frames:
                print("✗ 没有收集到有效的深度帧")
                return None, None
            
            # 多帧平均
            print(f"  对 {len(depth_frames)} 帧进行平均...")
            depth_frames = np.array(depth_frames)
            
            # 使用中值滤波去除异常值，然后平均
            depth_median = np.median(depth_frames, axis=0)
            depth_mean = np.mean(depth_frames, axis=0)
            
            # 结合中值和均值，提高稳定性
            depth_image = 0.7 * depth_median + 0.3 * depth_mean
            
            # 应用高斯滤波进一步平滑
            depth_image = cv2.GaussianBlur(depth_image, (5, 5), 0)
            
            print(f"✓ 多帧平均完成，最终深度范围: {depth_image.min():.2f} - {depth_image.max():.2f} mm")
            
            return rgb_image, depth_image
            
        except Exception as e:
            print(f"✗ 多帧深度图像捕获失败: {e}")
            return None, None
    
    def detect_objects(self, rgb_image):
        """
        使用YOLO检测物体
        
        Args:
            rgb_image (numpy.ndarray): RGB图像
            
        Returns:
            list: 检测结果列表，每个元素包含bbox、confidence、class_name等信息
        """
        if self.yolo_model is None:
            print("✗ YOLO模型未初始化")
            return []
        
        try:
            print(f"正在使用YOLO检测 {self.target_class} 物体...")
            
            # 运行检测
            results = self.yolo_model(rgb_image, verbose=False)
            
            # 处理检测结果
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        # 获取边界框坐标
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                        
                        # 获取置信度
                        confidence = boxes.conf[i].cpu().numpy()
                        
                        # 获取类别ID和名称
                        class_id = int(boxes.cls[i].cpu().numpy())
                        class_name = self.yolo_model.names[class_id].lower()
                        
                        # 只保留目标类别的检测结果
                        if class_name == self.target_class.lower() and confidence > 0.5:
                            detection = {
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(confidence),
                                'class_name': class_name,
                                'class_id': class_id
                            }
                            detections.append(detection)
                            print(f"✓ 检测到 {class_name}: 置信度={confidence:.3f}, "
                                  f"位置=[{int(x1)},{int(y1)},{int(x2)},{int(y2)}]")
            
            print(f"✓ 检测完成，共检测到 {len(detections)} 个 {self.target_class} 物体")
            return detections
            
        except Exception as e:
            print(f"✗ YOLO检测失败: {e}")
            return []
    
    def calculate_center_point(self, detection, depth_image):
        """
        计算物体中心点坐标和深度（改进版：区域平均+滤波）
        
        Args:
            detection (dict): 检测结果
            depth_image (numpy.ndarray): 深度图像
            
        Returns:
            tuple: (center_x, center_y, depth) 或 (None, None, None) 如果失败
        """
        try:
            # 获取边界框坐标
            bbox = detection['bbox']  # [x1, y1, x2, y2]
            
            # 计算中心点坐标
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)
            
            # 确保坐标在图像范围内
            if (center_x < 0 or center_x >= depth_image.shape[1] or 
                center_y < 0 or center_y >= depth_image.shape[0]):
                print(f"✗ 中心点坐标超出图像范围: ({center_x}, {center_y})")
                return None, None, None
            
            # 计算边界框区域内的稳定深度值
            center_depth = self.calculate_stable_depth(depth_image, center_x, center_y, bbox)
            
            if center_depth is None:
                print(f"✗ 无法获取有效的深度值")
                return None, None, None
            
            print(f"✓ 中心点计算完成: ({center_x}, {center_y}), 深度: {center_depth:.2f}mm")
            return center_x, center_y, center_depth
            
        except Exception as e:
            print(f"✗ 中心点计算失败: {e}")
            return None, None, None
    
    def calculate_stable_depth(self, depth_image, center_x, center_y, bbox, window_size=15):
        """
        计算中心点周围区域的稳定深度值
        
        Args:
            depth_image (numpy.ndarray): 深度图像
            center_x (int): 中心点x坐标
            center_y (int): 中心点y坐标
            bbox (list): 边界框 [x1, y1, x2, y2]
            window_size (int): 采样窗口大小
            
        Returns:
            float: 稳定的深度值，如果失败返回None
        """
        try:
            # 计算采样区域（在边界框内，以中心点为中心）
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            
            # 使用较小的窗口，但至少5x5
            actual_window = min(window_size, min(bbox_width, bbox_height) // 2)
            actual_window = max(actual_window, 5)
            
            # 计算采样区域边界
            x_start = max(center_x - actual_window // 2, 0)
            x_end = min(center_x + actual_window // 2 + 1, depth_image.shape[1])
            y_start = max(center_y - actual_window // 2, 0)
            y_end = min(center_y + actual_window // 2 + 1, depth_image.shape[0])
            
            # 提取深度区域
            depth_region = depth_image[y_start:y_end, x_start:x_end]
            
            # 过滤无效深度值（0值）
            valid_depths = depth_region[depth_region > 0]
            
            if len(valid_depths) == 0:
                print(f"  警告：中心点周围区域没有有效深度值")
                return None
            
            # 计算有效深度的统计信息
            depth_mean = np.mean(valid_depths)
            depth_std = np.std(valid_depths)
            depth_median = np.median(valid_depths)
            
            # 去除异常值（超过2个标准差的值）
            valid_mask = np.abs(valid_depths - depth_median) <= 2 * depth_std
            filtered_depths = valid_depths[valid_mask]
            
            if len(filtered_depths) == 0:
                print(f"  警告：过滤后没有有效深度值，使用原始中值")
                filtered_depths = valid_depths
            
            # 使用加权平均：中值权重更高
            final_depth = 0.7 * np.median(filtered_depths) + 0.3 * np.mean(filtered_depths)
            
            print(f"  深度统计: 均值={depth_mean:.2f}, 中值={depth_median:.2f}, "
                  f"标准差={depth_std:.2f}, 最终值={final_depth:.2f}mm")
            
            return final_depth
            
        except Exception as e:
            print(f"  深度计算失败: {e}")
            return None
    
    def save_grasp_point(self, center_x, center_y, depth):
        """
        保存抓取点信息到文件
        
        Args:
            center_x (int): 中心点x坐标
            center_y (int): 中心点y坐标
            depth (float): 深度值(mm)
        """
        try:
            # 确保data目录存在
            data_dir = os.path.dirname(self.point_file)
            os.makedirs(data_dir, exist_ok=True)
            
            # 保存抓取点信息到point.txt
            with open(self.point_file, 'w') as f:
                f.write(f"{center_x}\n")      # x坐标
                f.write(f"{center_y}\n")      # y坐标  
                f.write(f"{depth}\n")         # 深度值(mm)
            
            print(f"✓ 抓取点信息已保存到: {self.point_file}")
            print(f"  格式: x坐标={center_x}, y坐标={center_y}, 深度={depth:.2f}mm")
            
        except Exception as e:
            print(f"✗ 保存抓取点信息失败: {e}")
    
    def draw_detection_results(self, image, detections, center_point):
        """
        在图像上绘制检测结果和中心点
        
        Args:
            image (numpy.ndarray): 输入图像
            detections (list): 检测结果列表
            center_point (tuple): 中心点坐标 (x, y)
            
        Returns:
            numpy.ndarray: 绘制了检测结果的图像
        """
        result_image = image.copy()
        
        # 绘制所有检测框
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # 绘制边界框
            cv2.rectangle(result_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # 绘制标签
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(result_image, (bbox[0], bbox[1] - label_size[1] - 10), 
                         (bbox[0] + label_size[0], bbox[1]), (0, 255, 0), -1)
            cv2.putText(result_image, label, (bbox[0], bbox[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # 绘制中心点
        if center_point:
            center_x, center_y = center_point
            cv2.circle(result_image, (center_x, center_y), 8, (255, 0, 0), -1)  # 蓝色实心圆
            cv2.circle(result_image, (center_x, center_y), 15, (255, 0, 0), 2)   # 蓝色圆圈
            
            # 绘制十字准星
            cv2.line(result_image, (center_x - 20, center_y), (center_x + 20, center_y), (255, 0, 0), 2)
            cv2.line(result_image, (center_x, center_y - 20), (center_x, center_y + 20), (255, 0, 0), 2)
            
            # 添加中心点标签
            cv2.putText(result_image, f"Center: ({center_x},{center_y})", 
                       (center_x + 25, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        return result_image
    
    def run_detection_and_grasping(self):
        """
        运行完整的检测和抓取点计算流程
        
        Returns:
            bool: 是否成功
        """
        print("=" * 60)
        print("开始物体检测和抓取点计算")
        print("=" * 60)
        
        # 1. 捕获并保存图像
        rgb_image, depth_image = self.capture_and_save_images()
        if rgb_image is None or depth_image is None:
            print("✗ 图像捕获失败，程序退出")
            return False
        
        # 2. 使用YOLO检测物体
        detections = self.detect_objects(rgb_image)
        if not detections:
            print(f"✗ 未检测到目标物体: {self.target_class}")
            return False
        
        # 3. 选择置信度最高的检测结果
        best_detection = max(detections, key=lambda x: x['confidence'])
        print(f"✓ 选择置信度最高的物体: {best_detection['confidence']:.3f}")
        
        # 4. 计算中心点坐标和深度
        center_x, center_y, depth = self.calculate_center_point(best_detection, depth_image)
        if center_x is None:
            print("✗ 中心点计算失败")
            return False
        
        # 5. 保存抓取点信息
        self.save_grasp_point(center_x, center_y, depth)
        
        # 6. 绘制检测结果并保存
        result_image = self.draw_detection_results(rgb_image, detections, (center_x, center_y))
        result_path = os.path.join(self.save_dir, 'detection_result.png')
        cv2.imwrite(result_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
        print(f"✓ 检测结果图像已保存到: {result_path}")
        
        print("=" * 60)
        print("物体检测和抓取点计算完成!")
        print("=" * 60)
        
        return True

def main():
    """主函数"""
    print("物体检测和抓取点计算程序")
    print("此程序将检测指定物体并计算抓取点")
    
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
    
    # 创建检测器实例
    detector = ObjectDetectionAndGrasping(target_class="bottle")
    
    # 运行检测和抓取点计算
    success = detector.run_detection_and_grasping()
    
    if success:
        print("\n✓ 程序执行成功!")
    else:
        print("\n✗ 程序执行失败!")

if __name__ == "__main__":
    main() 