# yolo_detector.py
import cv2
import numpy as np
import os
from typing import List, Dict, Tuple, Optional
import torch

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("警告: ultralytics未安装，请运行: pip install ultralytics")

class YOLODetector:
    """
    YOLO物体检测器，集成到机械臂抓取系统中
    """
    
    def __init__(self, model_path: str = "yolov8n.pt", 
                 class_names: List[str] = None,
                 conf_threshold: float = 0.5,
                 nms_threshold: float = 0.4):
        """
        初始化YOLO检测器
        
        Args:
            model_path: YOLO模型路径，可以是预训练模型或自定义模型
            class_names: 目标类别名称列表
            conf_threshold: 置信度阈值
            nms_threshold: NMS阈值
        """
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.model = None
        self.model_path = model_path
        self.class_names = class_names
        
        if YOLO_AVAILABLE:
            self._load_model()
        else:
            print("YOLO模型未加载，请先安装ultralytics")
    
    def _load_model(self):
        """加载YOLO模型"""
        try:
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                print(f"YOLO模型加载成功: {self.model_path}")
            else:
                # 如果本地没有模型，自动下载
                self.model = YOLO(self.model_path)
                print(f"YOLO模型下载并加载成功: {self.model_path}")
        except Exception as e:
            print(f"YOLO模型加载失败: {e}")
            self.model = None
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        检测图像中的物体 (与grasper_detector.py兼容的方法)
        
        Args:
            image: 输入的RGB图像数组
            
        Returns:
            检测结果列表，每个元素包含边界框、置信度、类别等信息
        """
        return self.detect_objects_from_array(image)
    
    def detect_objects(self, image_path: str) -> List[Dict]:
        """
        检测图像中的物体
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            检测结果列表，每个元素包含边界框、置信度、类别等信息
        """
        if self.model is None:
            print("YOLO模型未加载")
            return []
        
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图像: {image_path}")
                return []
            
            # 运行检测
            results = self.model(image, conf=self.conf_threshold)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # 获取边界框坐标
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.model.names[class_id]
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': class_name,
                            'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                        }
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"检测过程中出现错误: {e}")
            return []
    
    def detect_objects_from_array(self, image_array: np.ndarray) -> List[Dict]:
        """
        从numpy数组检测物体
        
        Args:
            image_array: RGB图像数组
            
        Returns:
            检测结果列表
        """
        if self.model is None:
            print("YOLO模型未加载")
            return []
        
        try:
            # 运行检测
            results = self.model(image_array, conf=self.conf_threshold)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # 获取边界框坐标
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.model.names[class_id]
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': class_name,
                            'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                        }
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"检测过程中出现错误: {e}")
            return []
    
    def filter_detections_by_class(self, detections: List[Dict], target_classes: List[str]) -> List[Dict]:
        """
        根据目标类别过滤检测结果
        
        Args:
            detections: 检测结果列表
            target_classes: 目标类别名称列表
            
        Returns:
            过滤后的检测结果
        """
        filtered = []
        for detection in detections:
            if detection['class_name'] in target_classes:
                filtered.append(detection)
        return filtered
    
    def get_best_detection(self, detections: List[Dict], target_class: str = None) -> Optional[Dict]:
        """
        获取最佳检测结果（置信度最高）
        
        Args:
            detections: 检测结果列表
            target_class: 目标类别，如果为None则选择所有类别中置信度最高的
            
        Returns:
            最佳检测结果，如果没有则返回None
        """
        if not detections:
            return None
        
        if target_class:
            # 过滤指定类别
            filtered = self.filter_detections_by_class(detections, [target_class])
            if not filtered:
                return None
            detections = filtered
        
        # 选择置信度最高的
        best_detection = max(detections, key=lambda x: x['confidence'])
        return best_detection
    
    def draw_detections(self, image_path: str, detections: List[Dict], output_path: str = None) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        Args:
            image_path: 原始图像路径
            detections: 检测结果列表
            output_path: 输出图像路径，如果为None则不保存
            
        Returns:
            绘制了检测结果的图像数组
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return None
        
        # 绘制每个检测结果
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制标签
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # 绘制中心点
            center_x, center_y = detection['center']
            cv2.circle(image, (center_x, center_y), 3, (0, 0, 255), -1)
        
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"检测结果图像已保存到: {output_path}")
        
        return image

# 测试函数
def test_yolo_detector():
    """测试YOLO检测器"""
    detector = YOLODetector()
    
    # 测试图像路径（使用您现有的图像）
    test_image_path = "data/pose0/rgb2.jpg"
    
    if os.path.exists(test_image_path):
        print("开始检测...")
        detections = detector.detect_objects(test_image_path)
        
        print(f"检测到 {len(detections)} 个物体:")
        for i, detection in enumerate(detections):
            print(f"  {i+1}. {detection['class_name']} (置信度: {detection['confidence']:.2f})")
        
        # 绘制检测结果
        output_path = "data/pose0/detection_result.png"
        detector.draw_detections(test_image_path, detections, output_path)
        
    else:
        print(f"测试图像不存在: {test_image_path}")

if __name__ == "__main__":
    test_yolo_detector()