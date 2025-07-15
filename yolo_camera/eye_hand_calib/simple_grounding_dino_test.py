#!/usr/bin/env python3
"""
简化版GroundingDINO检测脚本
用于快速测试GroundingDINO功能，无需相机
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
import argparse

def load_grounding_dino_model():
    """加载GroundingDINO模型"""
    try:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        
        print("正在加载GroundingDINO模型...")
        model_name = "IDEA-Research/grounding-dino-base"
        
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
        
        # 检查是否可以使用GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        print(f"✓ 模型加载成功，使用设备: {device}")
        return processor, model, device
        
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return None, None, None

def detect_objects(processor, model, device, image_path, text_prompt, confidence_threshold=0.5):
    """检测物体"""
    try:
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        print(f"✓ 图像加载成功，尺寸: {image.size}")
        
        # 预处理
        inputs = processor(images=image, text=text_prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 推理
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 后处理
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_grounded_object_detection(
            outputs, target_sizes=target_sizes, threshold=confidence_threshold
        )
        
        # 解析结果
        detections = []
        for result in results:
            boxes = result["boxes"]
            scores = result["scores"]
            labels = result["labels"]
            
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box.cpu().numpy()
                confidence = score.cpu().numpy()
                
                detection = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(confidence),
                    'label': label
                }
                detections.append(detection)
        
        print(f"✓ 检测完成，共检测到 {len(detections)} 个物体")
        return detections, image
        
    except Exception as e:
        print(f"✗ 检测失败: {e}")
        return [], None

def draw_results(image, detections, output_path):
    """绘制检测结果"""
    try:
        # 转换为OpenCV格式
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 绘制检测框
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            label = detection['label']
            
            # 绘制边界框
            cv2.rectangle(image_cv, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # 绘制标签
            label_text = f"{label}: {confidence:.2f}"
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # 标签背景
            cv2.rectangle(image_cv, (bbox[0], bbox[1] - label_size[1] - 10), 
                         (bbox[0] + label_size[0], bbox[1]), (0, 255, 0), -1)
            
            # 标签文字
            cv2.putText(image_cv, label_text, (bbox[0], bbox[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # 中心点
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            cv2.circle(image_cv, (center_x, center_y), 5, (255, 0, 0), -1)
            
            print(f"  - {label}: 置信度={confidence:.3f}, 中心点=({center_x}, {center_y})")
        
        # 保存结果
        cv2.imwrite(output_path, image_cv)
        print(f"✓ 结果已保存到: {output_path}")
        
    except Exception as e:
        print(f"✗ 绘制结果失败: {e}")

def create_test_image(output_path):
    """创建测试图像"""
    try:
        # 创建白色背景
        image = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        # 绘制一些简单的形状作为测试物体
        # 红色矩形 (bottle)
        cv2.rectangle(image, (200, 150), (300, 350), (0, 0, 255), -1)
        
        # 蓝色圆形 (cup)
        cv2.circle(image, (450, 200), 50, (255, 0, 0), -1)
        
        # 绿色矩形 (phone)
        cv2.rectangle(image, (100, 300), (180, 400), (0, 255, 0), -1)
        
        # 保存图像
        cv2.imwrite(output_path, image)
        print(f"✓ 测试图像已创建: {output_path}")
        
    except Exception as e:
        print(f"✗ 创建测试图像失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="GroundingDINO物体检测测试")
    parser.add_argument("--image", type=str, help="输入图像路径")
    parser.add_argument("--text", type=str, default="bottle", help="检测文本描述")
    parser.add_argument("--threshold", type=float, default=0.3, help="置信度阈值")
    parser.add_argument("--output", type=str, default="detection_result.jpg", help="输出图像路径")
    parser.add_argument("--create-test", action="store_true", help="创建测试图像")
    
    args = parser.parse_args()
    
    print("GroundingDINO物体检测测试")
    print("=" * 50)
    
    # 创建测试图像
    if args.create_test:
        test_image_path = "test_image.jpg"
        create_test_image(test_image_path)
        args.image = test_image_path
    
    # 检查输入图像
    if not args.image:
        print("错误: 请指定输入图像路径或使用 --create-test 创建测试图像")
        return
    
    if not os.path.exists(args.image):
        print(f"错误: 图像文件不存在: {args.image}")
        return
    
    # 加载模型
    processor, model, device = load_grounding_dino_model()
    if processor is None:
        print("错误: 模型加载失败")
        return
    
    # 检测物体
    print(f"正在检测物体: '{args.text}'")
    print(f"置信度阈值: {args.threshold}")
    
    detections, image = detect_objects(
        processor, model, device, 
        args.image, args.text, args.threshold
    )
    
    if detections:
        # 绘制结果
        draw_results(image, detections, args.output)
        
        # 保存检测信息
        info_path = args.output.replace('.jpg', '_info.txt')
        with open(info_path, 'w') as f:
            f.write(f"检测目标: {args.text}\n")
            f.write(f"置信度阈值: {args.threshold}\n")
            f.write(f"检测到的物体数量: {len(detections)}\n\n")
            
            for i, detection in enumerate(detections):
                bbox = detection['bbox']
                center_x = (bbox[0] + bbox[2]) // 2
                center_y = (bbox[1] + bbox[3]) // 2
                
                f.write(f"物体 {i+1}:\n")
                f.write(f"  标签: {detection['label']}\n")
                f.write(f"  置信度: {detection['confidence']:.3f}\n")
                f.write(f"  边界框: {bbox}\n")
                f.write(f"  中心点: ({center_x}, {center_y})\n\n")
        
        print(f"✓ 检测信息已保存到: {info_path}")
        print("✓ 检测完成!")
        
    else:
        print("未检测到任何物体")

if __name__ == "__main__":
    main()
