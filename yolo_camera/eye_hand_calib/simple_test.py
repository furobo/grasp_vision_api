#!/usr/bin/env python3
"""
简单测试脚本：演示YOLO物体检测和抓取点计算功能
"""

import sys
import os

# 添加模块路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def simple_test():
    """简单测试YOLO检测功能"""
    print("开始测试YOLO物体检测功能...")
    
    try:
        # 导入必要的模块
        from calibration_viewer import CameraWindow
        from PyQt5.QtWidgets import QApplication
        
        # 创建QApplication（必须在创建任何Qt部件之前）
        app = QApplication(sys.argv)
        
        # 创建CameraWindow实例
        print("正在初始化相机系统...")
        camera_window = CameraWindow()
        
        # 测试检测功能
        print("开始检测bottle物体...")
        success = camera_window.detect_and_grasp_object("bottle")
        
        if success:
            print("\n✓ 检测成功！")
            print("请查看以下目录的结果文件:")
            print("  /home/adminpc/Documents/复旦复合机器人/手眼标定demo/robotics_v20250704/robotics/eye_hand_calib/data/yolo_detections/")
            
            # 读取并显示结果
            result_dir = "/home/adminpc/Documents/复旦复合机器人/手眼标定demo/robotics_v20250704/robotics/eye_hand_calib/data/yolo_detections"
            point_file = os.path.join(result_dir, "point.txt")
            
            if os.path.exists(point_file):
                print("\n抓取点坐标信息:")
                with open(point_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) >= 3:
                        x = int(float(lines[0].strip()))
                        y = int(float(lines[1].strip()))
                        depth = float(lines[2].strip())
                        print(f"  X坐标: {x} 像素")
                        print(f"  Y坐标: {y} 像素")
                        print(f"  深度: {depth:.2f} mm ({depth/1000:.3f} m)")
        else:
            print("\n✗ 未检测到bottle物体")
            print("请确保:")
            print("  1. 场景中有bottle物体")
            print("  2. 相机连接正常")
            print("  3. 光照条件良好")
            
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_test() 