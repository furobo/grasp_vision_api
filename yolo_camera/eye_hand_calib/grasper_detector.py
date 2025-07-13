#!/usr/bin/env python3
"""
YOLO Grasp Detector Module
Integrates YOLO object detection with 3D position calculation for robotic grasping
"""

import numpy as np
import cv2
from yolo_detector import YOLODetector
import yaml
import os

class GraspDetector:
    """
    Grasp detector that combines YOLO object detection with 3D position calculation
    """
    
    def __init__(self, config_path="config.yaml"):
        """
        Initialize the grasp detector
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize YOLO detector
        self.yolo_detector = YOLODetector(
            model_path=self.config['yolo']['model_path'],
            class_names=self.config['yolo']['class_names'],
            conf_threshold=self.config['yolo']['conf_threshold'],
            nms_threshold=self.config['yolo']['nms_threshold']
        )
        
        # Camera intrinsic matrix (will be loaded from calibration)
        self.intrinsic_matrix = None
        self.load_intrinsic_matrix()
    
    def load_intrinsic_matrix(self):
        """Load camera intrinsic matrix from calibration file"""
        intrinsic_path = "data/intrinsic.txt"
        if os.path.exists(intrinsic_path):
            self.intrinsic_matrix = np.loadtxt(intrinsic_path)
            print(f"Loaded intrinsic matrix from {intrinsic_path}")
        else:
            print(f"Warning: Intrinsic matrix file not found at {intrinsic_path}")
    
    def detect_objects(self, rgb_image, depth_image):
        """
        Detect objects in the image and calculate their 3D positions
        
        Args:
            rgb_image: RGB image from camera
            depth_image: Depth image from camera
            
        Returns:
            list: List of detected objects with 3D positions
        """
        # Run YOLO detection
        detections = self.yolo_detector.detect(rgb_image)
        
        # Calculate 3D positions for detected objects
        grasp_candidates = []
        
        for detection in detections:
            # Extract bounding box
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Calculate center point of bounding box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Get depth value at center point
            if (0 <= center_x < depth_image.shape[1] and 
                0 <= center_y < depth_image.shape[0]):
                
                depth_value = depth_image[center_y, center_x]
                
                # Convert to 3D position using camera intrinsics
                if self.intrinsic_matrix is not None and depth_value > 0:
                    # Convert pixel coordinates to 3D coordinates
                    fx = self.intrinsic_matrix[0, 0]
                    fy = self.intrinsic_matrix[1, 1]
                    cx = self.intrinsic_matrix[0, 2]
                    cy = self.intrinsic_matrix[1, 2]
                    
                    # Calculate 3D position in camera frame
                    z = depth_value / 1000.0  # Convert mm to meters
                    x = (center_x - cx) * z / fx
                    y = (center_y - cy) * z / fy
                    
                    grasp_candidate = {
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2],
                        'center_2d': [center_x, center_y],
                        'position_3d': [x, y, z],
                        'depth': depth_value
                    }
                    
                    grasp_candidates.append(grasp_candidate)
        
        return grasp_candidates
    
    def get_best_grasp_target(self, grasp_candidates):
        """
        Select the best grasp target from candidates
        
        Args:
            grasp_candidates: List of grasp candidates
            
        Returns:
            dict: Best grasp target or None if no valid target
        """
        if not grasp_candidates:
            return None
        
        # Sort by confidence (highest first)
        sorted_candidates = sorted(grasp_candidates, 
                                 key=lambda x: x['confidence'], 
                                 reverse=True)
        
        # Return the highest confidence detection
        return sorted_candidates[0]
    
    def visualize_detections(self, rgb_image, grasp_candidates):
        """
        Visualize detected objects on the image
        
        Args:
            rgb_image: RGB image
            grasp_candidates: List of grasp candidates
            
        Returns:
            np.ndarray: Image with visualized detections
        """
        vis_image = rgb_image.copy()
        
        for candidate in grasp_candidates:
            # Draw bounding box
            x1, y1, x2, y2 = candidate['bbox']
            cv2.rectangle(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Draw center point
            center_x, center_y = candidate['center_2d']
            cv2.circle(vis_image, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # Draw label
            label = f"{candidate['class_name']}: {candidate['confidence']:.2f}"
            if candidate['position_3d']:
                x, y, z = candidate['position_3d']
                label += f" ({x:.2f}, {y:.2f}, {z:.2f})"
            
            cv2.putText(vis_image, label, (int(x1), int(y1-10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return vis_image

# Example usage
if __name__ == "__main__":
    # Test the grasp detector
    detector = GraspDetector()
    print("Grasp detector initialized successfully!")
    print(f"YOLO model: {detector.config['yolo']['model_path']}")
    print(f"Target classes: {detector.config['yolo']['class_names']}") 