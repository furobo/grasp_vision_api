# grasp_detector.py
import numpy as np
from typing import List, Dict, Optional, Tuple
from yolo_detector import YOLODetector
from world_image import WorldImage

class GraspDetector:
    """
    集成YOLO检测和3D位置计算的抓取检测器
    """
    
    def __init__(self, yolo_model_path: str = "yolov8n.pt", conf_threshold: float = 0.5):
        """
        初始化抓取检测器
        
        Args:
            yolo_model_path: YOLO模型路径
            conf_threshold: 置信度阈值
        """
        self.yolo_detector = YOLODetector(yolo_model_path, conf_threshold)
        self.world_image = None
        
    def set_world_image(self, world_image: WorldImage):
        """设置世界图像实例"""
        self.world_image = world_image
    
    def detect_grasp_targets(self, image_path: str, target_classes: List[str] = None) -> List[Dict]:
        """
        检测可抓取的目标物体
        
        Args:
            image_path: 图像路径
            target_classes: 目标类别列表，如果为None则检测所有类别
            
        Returns:
            包含3D位置信息的检测结果列表
        """
        if self.world_image is None:
            print("警告: 未设置world_image实例，无法获取3D位置信息")
            return []
        
        # 1. YOLO检测
        detections = self.yolo_detector.detect_objects(image_path)
        
        if not detections:
            return []
        
        # 2. 过滤目标类别
        if target_classes:
            detections = self.yolo_detector.filter_detections_by_class(detections, target_classes)
        
        # 3. 添加3D位置信息
        grasp_targets = []
        for detection in detections:
            center_x, center_y = detection['center']
            
            # 获取3D位置（基坐标系）
            base_position = self.world_image.get_point_position(
                pixel_row=center_y,
                pixel_col=center_x,
                frame_name='base'
            )
            
            # 检查深度是否有效
            if np.any(base_position != 0):
                grasp_target = detection.copy()
                grasp_target['base_position'] = base_position
                grasp_target['pixel_position'] = [center_x, center_y]
                grasp_targets.append(grasp_target)
            else:
                print(f"警告: 物体 {detection['class_name']} 的深度信息无效")
        
        return grasp_targets
    
    def select_best_grasp_target(self, grasp_targets: List[Dict], 
                                target_class: str = None,
                                min_confidence: float = 0.5) -> Optional[Dict]:
        """
        选择最佳抓取目标
        
        Args:
            grasp_targets: 抓取目标列表
            target_class: 目标类别
            min_confidence: 最小置信度
            
        Returns:
            最佳抓取目标
        """
        if not grasp_targets:
            return None
        
        # 过滤置信度
        valid_targets = [t for t in grasp_targets if t['confidence'] >= min_confidence]
        
        if not valid_targets:
            return None
        
        # 如果指定了目标类别，优先选择该类别
        if target_class:
            class_targets = [t for t in valid_targets if t['class_name'] == target_class]
            if class_targets:
                valid_targets = class_targets
        
        # 选择置信度最高的
        best_target = max(valid_targets, key=lambda x: x['confidence'])
        return best_target
    
    def calculate_grasp_pose(self, grasp_target: Dict, 
                           approach_distance: float = 100.0,
                           lift_distance: float = 150.0) -> Dict:
        """
        计算抓取姿态
        
        Args:
            grasp_target: 抓取目标信息
            approach_distance: 预抓取距离（mm）
            lift_distance: 提升距离（mm）
            
        Returns:
            抓取姿态信息
        """
        base_position = grasp_target['base_position']
        x, y, z = base_position
        
        # 基础抓取位置
        grasp_position = [x, y, z]
        
        # 预抓取位置（上方）
        pre_grasp_position = [x, y, z + approach_distance]
        
        # 提升位置
        lift_position = [x, y, z + lift_distance]
        
        # 抓取姿态（从上方垂直抓取）
        grasp_orientation = [180, 0, 0]  # 欧拉角（度）
        
        grasp_pose = {
            'target_info': grasp_target,
            'grasp_position': grasp_position,
            'pre_grasp_position': pre_grasp_position,
            'lift_position': lift_position,
            'grasp_orientation': grasp_orientation,
            'confidence': grasp_target['confidence']
        }
        
        return grasp_pose
    
    def visualize_grasp_targets(self, image_path: str, grasp_targets: List[Dict], 
                              output_path: str = None) -> np.ndarray:
        """
        可视化抓取目标
        
        Args:
            image_path: 原始图像路径
            grasp_targets: 抓取目标列表
            output_path: 输出路径
            
        Returns:
            可视化图像
        """
        # 转换为YOLO检测格式
        detections = []
        for target in grasp_targets:
            detection = {
                'bbox': target['bbox'],
                'confidence': target['confidence'],
                'class_name': target['class_name'],
                'center': target['pixel_position']
            }
            detections.append(detection)
        
        # 使用YOLO检测器的绘制功能
        return self.yolo_detector.draw_detections(image_path, detections, output_path)

# 测试函数
def test_grasp_detector():
    """测试抓取检测器"""
    from world_image import WorldImage
    
    # 创建抓取检测器
    grasp_detector = GraspDetector()
    
    # 设置世界图像实例
    world_image = WorldImage()
    grasp_detector.set_world_image(world_image)
    
    # 测试图像路径
    test_image_path = "data/pose0/rgb.png"
    
    if os.path.exists(test_image_path):
        print("开始抓取目标检测...")
        
        # 检测可抓取的目标（以瓶子为例）
        grasp_targets = grasp_detector.detect_grasp_targets(
            test_image_path, 
            target_classes=['bottle', 'cup', 'bowl']
        )
        
        print(f"检测到 {len(grasp_targets)} 个可抓取目标:")
        for i, target in enumerate(grasp_targets):
            pos = target['base_position']
            print(f"  {i+1}. {target['class_name']} - 位置: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}] mm")
        
        # 选择最佳目标
        best_target = grasp_detector.select_best_grasp_target(grasp_targets, target_class='bottle')
        
        if best_target:
            print(f"\n最佳抓取目标: {best_target['class_name']}")
            
            # 计算抓取姿态
            grasp_pose = grasp_detector.calculate_grasp_pose(best_target)
            print(f"抓取位置: {grasp_pose['grasp_position']}")
            print(f"预抓取位置: {grasp_pose['pre_grasp_position']}")
        
        # 可视化结果
        output_path = "data/pose0/grasp_targets.png"
        grasp_detector.visualize_grasp_targets(test_image_path, grasp_targets, output_path)
        
    else:
        print(f"测试图像不存在: {test_image_path}")

if __name__ == "__main__":
    import os
    test_grasp_detector()