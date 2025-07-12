#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视觉抓取系统使用示例
展示如何在实际项目中使用视觉抓取系统
"""

import numpy as np
from vision_based_pick_place import create_vision_pick_system, execute_vision_pick, calculate_object_position

def example_1_basic_usage():
    """
    示例1: 基本使用方法
    """
    print("=== 示例1: 基本使用方法 ===")
    
    # 创建视觉抓取系统
    vision_system = create_vision_pick_system()
    
    # 模拟从视觉系统获取的数据
    distance = 350.0  # mm
    pixel_u = 680.0   # 像素坐标
    pixel_v = 320.0   # 像素坐标
    
    print(f"检测到物体: 距离={distance}mm, 像素坐标=({pixel_u}, {pixel_v})")
    
    # 执行抓取
    success = execute_vision_pick(vision_system, distance, pixel_u, pixel_v)
    
    if success:
        print("✅ 抓取成功!")
    else:
        print("❌ 抓取失败!")

def example_2_position_calculation_only():
    """
    示例2: 仅计算物体位置，不执行抓取
    """
    print("\n=== 示例2: 仅计算物体位置 ===")
    
    # 创建视觉抓取系统
    vision_system = create_vision_pick_system()
    
    # 多个物体的检测数据
    objects = [
        {"distance": 300.0, "u": 640.0, "v": 360.0, "name": "物体1"},
        {"distance": 280.0, "u": 720.0, "v": 300.0, "name": "物体2"},
        {"distance": 320.0, "u": 560.0, "v": 400.0, "name": "物体3"}
    ]
    
    # 计算每个物体的位置
    for obj in objects:
        position = calculate_object_position(
            vision_system, 
            obj["distance"], 
            obj["u"], 
            obj["v"]
        )
        print(f"{obj['name']}: 基座坐标 = [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}] mm")

def example_3_batch_processing():
    """
    示例3: 批量处理多个物体
    """
    print("\n=== 示例3: 批量处理多个物体 ===")
    
    # 创建视觉抓取系统
    vision_system = create_vision_pick_system()
    
    # 批量物体数据
    objects_to_pick = [
        {"distance": 300.0, "u": 650.0, "v": 350.0, "name": "红色方块"},
        {"distance": 280.0, "u": 700.0, "v": 320.0, "name": "蓝色圆柱"},
        {"distance": 320.0, "u": 580.0, "v": 380.0, "name": "绿色三角"}
    ]
    
    success_count = 0
    
    for i, obj in enumerate(objects_to_pick):
        print(f"\n正在处理第{i+1}个物体: {obj['name']}")
        
        # 计算位置
        position = calculate_object_position(
            vision_system, 
            obj["distance"], 
            obj["u"], 
            obj["v"]
        )
        
        print(f"计算位置: [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}] mm")
        
        # 检查位置是否合理
        if is_position_safe(position):
            print("位置安全，开始抓取...")
            success = execute_vision_pick(vision_system, obj["distance"], obj["u"], obj["v"])
            if success:
                success_count += 1
                print(f"✅ {obj['name']} 抓取成功!")
            else:
                print(f"❌ {obj['name']} 抓取失败!")
        else:
            print(f"⚠️  {obj['name']} 位置不安全，跳过抓取")
    
    print(f"\n批量处理完成，成功抓取 {success_count}/{len(objects_to_pick)} 个物体")

def is_position_safe(position):
    """
    检查位置是否安全
    
    Args:
        position: 物体位置坐标
    
    Returns:
        bool: 位置是否安全
    """
    # 简单的安全检查（根据实际机械臂工作空间调整）
    x, y, z = position
    
    # 检查是否在工作空间内
    if abs(x) > 800 or abs(y) > 800 or z < 50 or z > 600:
        return False
    
    # 检查是否避开危险区域
    if x > 300 and x < 400 and y > -100 and y < 100:
        return False  # 假设这是一个危险区域
    
    return True

def example_4_real_time_simulation():
    """
    示例4: 模拟实时视觉抓取
    """
    print("\n=== 示例4: 模拟实时视觉抓取 ===")
    
    # 创建视觉抓取系统
    vision_system = create_vision_pick_system()
    
    # 模拟连续的视觉检测数据
    detection_sequence = [
        {"distance": 300.0, "u": 640.0, "v": 360.0, "timestamp": 1.0},
        {"distance": 295.0, "u": 642.0, "v": 358.0, "timestamp": 1.1},
        {"distance": 290.0, "u": 645.0, "v": 355.0, "timestamp": 1.2},
        {"distance": 285.0, "u": 648.0, "v": 352.0, "timestamp": 1.3},
        {"distance": 280.0, "u": 650.0, "v": 350.0, "timestamp": 1.4}
    ]
    
    print("模拟物体移动的检测序列：")
    for detection in detection_sequence:
        position = calculate_object_position(
            vision_system, 
            detection["distance"], 
            detection["u"], 
            detection["v"]
        )
        print(f"时间 {detection['timestamp']}s: "
              f"像素({detection['u']}, {detection['v']}), "
              f"距离 {detection['distance']}mm, "
              f"基座坐标 [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}] mm")
    
    # 使用最后一个检测结果进行抓取
    final_detection = detection_sequence[-1]
    print(f"\n使用最终检测结果执行抓取...")
    success = execute_vision_pick(
        vision_system, 
        final_detection["distance"], 
        final_detection["u"], 
        final_detection["v"]
    )
    
    if success:
        print("✅ 实时抓取成功!")
    else:
        print("❌ 实时抓取失败!")

def main():
    """
    主函数：运行所有示例
    """
    print("=== 视觉抓取系统使用示例 ===")
    
    try:
        # 运行示例1
        example_1_basic_usage()
        
        # 运行示例2  
        example_2_position_calculation_only()
        
        # 运行示例3
        example_3_batch_processing()
        
        # 运行示例4
        example_4_real_time_simulation()
        
    except Exception as e:
        print(f"示例运行出错: {e}")
        print("请检查机械臂和夹爪连接状态")
    
    print("\n所有示例运行完成!")

if __name__ == '__main__':
    main()
