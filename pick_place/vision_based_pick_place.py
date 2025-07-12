#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于视觉的机械臂抓取程序
功能: 根据距离信息，计算物体在基座系下的坐标，并控制机械臂完成抓取任务
作者: AI Assistant
日期: 2025年7月12日
"""

import math
import sys
import os
import time
import numpy as np
import transforms3d as tfs
from simple_pick_place import SiasunRobotPythonInterface, DHGrasperInterface


class VisionBasedPickPlace:
    def __init__(self, robot_ip='192.168.5.111', gripper_port='/dev/ttyUSB0'):
        """
        初始化视觉抓取系统
        
        Args:
            robot_ip (str): 机械臂IP地址
            gripper_port (str): 夹爪串口
        """
        print("=== 初始化视觉抓取系统 ===")
        
        # 初始化机械臂
        self.robot = SiasunRobotPythonInterface(robot_ip)
        print("机械臂初始化完成")
        
        # 初始化夹爪
        self.gripper = DHGrasperInterface(gripper_port, 115200)
        self.gripper.set_speed(50)
        self.gripper.set_force(50)
        print("夹爪初始化完成")
        
        # 加载手眼标定矩阵
        self.eye2tcp_matrix = self.load_eye2tcp_matrix()
        print("手眼标定矩阵加载完成")
        print(f"Eye2TCP矩阵:\n{self.eye2tcp_matrix}")
        
        # 加载相机内参
        self.camera_intrinsics = self.load_camera_intrinsics()
        print("相机内参加载完成")
        print(f"相机内参: fx={self.camera_intrinsics['fx']}, fy={self.camera_intrinsics['fy']}, cx={self.camera_intrinsics['cx']}, cy={self.camera_intrinsics['cy']}")
        
        # 设置抓取参数
        self.grasp_height_offset = 50  # 抓取时的高度偏移量(mm)
        self.approach_height = 100     # 接近物体时的高度(mm)
        self.place_position = [500, 177, 200]  # 放置位置
        self.init_euler = [-90.01709163778153, 0.5653085040623331, -97.95937903864494]  # 初始化姿态
        
        print("视觉抓取系统初始化完成\n")
    
    def load_eye2tcp_matrix(self):
        """
        加载手眼标定矩阵
        
        Returns:
            np.ndarray: 4x4的手眼标定矩阵
        """
        try:
            matrix_file = '/Users/vagrant/Documents/try/tranform/eye2tcp_matrix.txt'
            matrix = np.loadtxt(matrix_file)
            print(f"成功加载手眼标定矩阵: {matrix_file}")
            return matrix
        except Exception as e:
            print(f"加载手眼标定矩阵失败: {e}")
            # 如果加载失败，返回单位矩阵
            return np.eye(4)
    
    def load_camera_intrinsics(self):
        """
        加载相机内参
        
        Returns:
            dict: 包含fx, fy, cx, cy的字典
        """
        try:
            intrinsics_file = '/Users/vagrant/Documents/try/tranform/intrinsic.txt'
            intrinsic_matrix = np.loadtxt(intrinsics_file)
            print(f"成功加载相机内参: {intrinsics_file}")
            
            # 从内参矩阵中提取参数
            fx = intrinsic_matrix[0, 0]
            fy = intrinsic_matrix[1, 1]
            cx = intrinsic_matrix[0, 2]
            cy = intrinsic_matrix[1, 2]
            
            return {
                'fx': fx,
                'fy': fy,
                'cx': cx,
                'cy': cy
            }
        except Exception as e:
            print(f"加载相机内参失败: {e}")
            # 如果加载失败，返回默认值
            return {
                'fx': 905.578125,
                'fy': 904.208557,
                'cx': 637.027466,
                'cy': 379.470856
            }
    
    def calculate_camera_pose_in_base(self):
        """
        计算相机在基座系下的位姿
        
        Returns:
            tuple: (相机在基座系下的位置, 相机在基座系下的姿态)
        """
        print("--- 计算相机在基座系下的位姿 ---")
        
        # 获取当前TCP(末端执行器)在基座系下的位姿
        tcp_pos, tcp_euler = self.robot.get_xyz_eulerdeg()
        print(f"当前TCP位置: {tcp_pos}")
        print(f"当前TCP姿态(欧拉角): {tcp_euler}")
        
        # 构建TCP在基座系下的变换矩阵
        tcp_pos_m = np.array(tcp_pos) / 1000.0  # 转换为米
        tcp_euler_rad = np.deg2rad(tcp_euler)   # 转换为弧度
        
        # 使用ZYX欧拉角顺序构建旋转矩阵
        tcp_rotation = tfs.euler.euler2mat(tcp_euler_rad[0], tcp_euler_rad[1], tcp_euler_rad[2], 'rzyx')
        
        # 构建TCP到基座的变换矩阵
        tcp2base_matrix = np.eye(4)
        tcp2base_matrix[:3, :3] = tcp_rotation
        tcp2base_matrix[:3, 3] = tcp_pos_m
        
        print(f"TCP到基座变换矩阵:\n{tcp2base_matrix}")
        
        # 计算相机在基座系下的变换矩阵
        # camera_in_base = tcp_in_base * eye2tcp
        camera2base_matrix = np.dot(tcp2base_matrix, self.eye2tcp_matrix)
        
        # 提取相机在基座系下的位置和姿态
        camera_pos = camera2base_matrix[:3, 3] * 1000  # 转换为毫米
        camera_rotation = camera2base_matrix[:3, :3]
        
        # 将旋转矩阵转换为欧拉角
        camera_euler_rad = tfs.euler.mat2euler(camera_rotation, 'rzyx')
        camera_euler = np.rad2deg(camera_euler_rad)
        
        print(f"相机在基座系下的位置: {camera_pos}")
        print(f"相机在基座系下的姿态: {camera_euler}")
        
        return camera_pos, camera_euler
    
    def calculate_object_position_in_base(self, distance, u, v, fx=None, fy=None, cx=None, cy=None):
        """
        根据距离信息、像素坐标和相机内参，计算物体在基座坐标系下的位置
        
        Args:
            distance (float): 相机到物体的距离 (mm)
            u (float): 物体在图像中的水平像素坐标
            v (float): 物体在图像中的垂直像素坐标
            fx (float, optional): 相机水平焦距 (像素)，如果为None则使用加载的内参
            fy (float, optional): 相机垂直焦距 (像素)，如果为None则使用加载的内参
            cx (float, optional): 相机主点水平坐标 (像素)，如果为None则使用加载的内参
            cy (float, optional): 相机主点垂直坐标 (像素)，如果为None则使用加载的内参
        
        Returns:
            np.ndarray: 物体在基座坐标系下的位置 [x, y, z] (mm)
        """
        print(f"--- 计算物体在基座系下的坐标 (距离: {distance}mm, 像素: ({u}, {v})) ---")
        
        # 使用传入的内参或加载的内参
        fx = fx if fx is not None else self.camera_intrinsics['fx']
        fy = fy if fy is not None else self.camera_intrinsics['fy']
        cx = cx if cx is not None else self.camera_intrinsics['cx']
        cy = cy if cy is not None else self.camera_intrinsics['cy']
        
        print(f"使用的相机内参: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
        
        # 根据你提供的公式计算物体在相机坐标系中的位置 (单位: mm)
        x_cam = (u - cx) * distance / fx
        y_cam = (v - cy) * distance / fy
        z_cam = distance
        
        print(f"物体在相机坐标系中的位置: [{x_cam:.2f}, {y_cam:.2f}, {z_cam:.2f}] mm")
        
        # 转换为齐次坐标，单位转换为米 (后续计算需要米)
        object_in_camera = np.array([x_cam / 1000.0, y_cam / 1000.0, z_cam / 1000.0, 1])
        
        # 获取TCP在基座系下的位姿
        tcp_pos, tcp_euler = self.robot.get_xyz_eulerdeg()
        tcp_pos_m = np.array(tcp_pos) / 1000.0  # 转换为米
        tcp_euler_rad = np.deg2rad(tcp_euler)
        
        print(f"当前TCP位置: {tcp_pos} mm")
        print(f"当前TCP姿态: {tcp_euler} 度")
        
        # 构建TCP到基座的旋转矩阵 (使用ZYX欧拉角顺序)
        tcp_rotation = tfs.euler.euler2mat(tcp_euler_rad[0], tcp_euler_rad[1], tcp_euler_rad[2], 'rzyx')
        
        # 构建TCP到基座的变换矩阵
        tcp2base_matrix = np.eye(4)
        tcp2base_matrix[:3, :3] = tcp_rotation
        tcp2base_matrix[:3, 3] = tcp_pos_m
        
        print(f"TCP到基座变换矩阵:\n{tcp2base_matrix}")
        
        # 计算相机到基座的变换矩阵
        # camera_in_base = tcp_in_base * eye2tcp
        camera2base_matrix = np.dot(tcp2base_matrix, self.eye2tcp_matrix)
        
        print(f"相机到基座变换矩阵:\n{camera2base_matrix}")
        
        # 将物体从相机坐标系转换到基座坐标系
        object_in_base = np.dot(camera2base_matrix, object_in_camera)
        object_pos_mm = object_in_base[:3] * 1000  # 转换回毫米
        
        print(f"物体在基座系下的位置: [{object_pos_mm[0]:.2f}, {object_pos_mm[1]:.2f}, {object_pos_mm[2]:.2f}] mm")
        
        return object_pos_mm
    
    def execute_grasp_sequence(self, target_position):
        """
        执行抓取序列
        
        Args:
            target_position (list): 目标物体的位置坐标[x, y, z]
        """
        print("=== 开始执行抓取序列 ===")
        
        # 步骤1: 初始化夹爪姿态
        print("步骤1: 初始化夹爪姿态")
        current_pos, current_euler = self.robot.get_xyz_eulerdeg()
        print(f"当前位置: {current_pos}")
        print(f"当前姿态: {current_euler}")
        print(f"目标姿态: {self.init_euler}")
        
        ret = self.robot.move_to_absolute_position(
            current_pos[0], current_pos[1], current_pos[2],
            self.init_euler[0], self.init_euler[1], self.init_euler[2],
            motion_type='joint'
        )
        print(f"姿态初始化结果: {ret}")
        time.sleep(2)
        
        # 步骤2: 打开夹爪
        print("步骤2: 打开夹爪")
        self.gripper.open_gripper()
        time.sleep(1)
        
        # 步骤3: 移动到物体上方的接近位置
        approach_pos = [target_position[0], target_position[1], target_position[2] + self.approach_height]
        print(f"步骤3: 移动到接近位置 {approach_pos}")
        ret = self.robot.move_to_absolute_position(
            approach_pos[0], approach_pos[1], approach_pos[2],
            self.init_euler[0], self.init_euler[1], self.init_euler[2],
            motion_type='joint'
        )
        print(f"移动到接近位置结果: {ret}")
        time.sleep(1)
        
        # 步骤4: 下降到抓取位置
        grasp_pos = [target_position[0], target_position[1], target_position[2] + self.grasp_height_offset]
        print(f"步骤4: 下降到抓取位置 {grasp_pos}")
        ret = self.robot.move_to_absolute_position(
            grasp_pos[0], grasp_pos[1], grasp_pos[2],
            self.init_euler[0], self.init_euler[1], self.init_euler[2],
            motion_type='linear'
        )
        print(f"下降到抓取位置结果: {ret}")
        time.sleep(1)
        
        # 步骤5: 关闭夹爪进行抓取
        print("步骤5: 关闭夹爪进行抓取")
        self.gripper.close_gripper()
        time.sleep(2)
        
        # 步骤6: 提升到安全高度
        print("步骤6: 提升到安全高度")
        ret = self.robot.move_to_absolute_position(
            grasp_pos[0], grasp_pos[1], grasp_pos[2] + 100,
            self.init_euler[0], self.init_euler[1], self.init_euler[2],
            motion_type='linear'
        )
        print(f"提升到安全高度结果: {ret}")
        time.sleep(1)
        
        # 步骤7: 移动到放置位置
        place_pos = [self.place_position[0], self.place_position[1], self.place_position[2] + 100]
        print(f"步骤7: 移动到放置位置上方 {place_pos}")
        ret = self.robot.move_to_absolute_position(
            place_pos[0], place_pos[1], place_pos[2],
            self.init_euler[0], self.init_euler[1], self.init_euler[2],
            motion_type='joint'
        )
        print(f"移动到放置位置上方结果: {ret}")
        time.sleep(1)
        
        # 步骤8: 下降到放置位置
        print(f"步骤8: 下降到放置位置 {self.place_position}")
        ret = self.robot.move_to_absolute_position(
            self.place_position[0], self.place_position[1], self.place_position[2],
            self.init_euler[0], self.init_euler[1], self.init_euler[2],
            motion_type='linear'
        )
        print(f"下降到放置位置结果: {ret}")
        time.sleep(1)
        
        # 步骤9: 打开夹爪放置物体
        print("步骤9: 打开夹爪放置物体")
        self.gripper.open_gripper()
        time.sleep(1)
        
        # 步骤10: 回到安全位置
        print("步骤10: 回到安全位置")
        ret = self.robot.move_to_safe_position()
        print(f"回到安全位置结果: {ret}")
        
        print("=== 抓取序列执行完成 ===")
    
    def vision_based_pick_and_place(self, distance, pixel_u, pixel_v):
        """
        基于视觉的抓取主函数
        
        Args:
            distance (float): 相机到物体中心的距离(mm)
            pixel_u (float): 物体在图像中的水平像素坐标
            pixel_v (float): 物体在图像中的垂直像素坐标
        """
        print(f"=== 开始基于视觉的抓取任务 ===")
        print(f"输入参数:")
        print(f"  - 距离: {distance}mm")
        print(f"  - 像素坐标: ({pixel_u}, {pixel_v})")
        
        try:
            # 计算物体在基座系下的位置
            object_position = self.calculate_object_position_in_base(distance, pixel_u, pixel_v)
            print(f"计算得到的物体位置: [{object_position[0]:.2f}, {object_position[1]:.2f}, {object_position[2]:.2f}] mm")
            
            # 执行抓取序列
            self.execute_grasp_sequence(object_position)
            
            print("抓取任务完成成功！")
            
        except Exception as e:
            print(f"抓取任务执行失败: {e}")
            print("尝试回到安全位置...")
            try:
                self.robot.move_to_safe_position()
            except:
                print("回到安全位置也失败了，请手动检查机械臂状态")
    
    def test_system(self):
        """
        测试系统功能
        """
        print("=== 系统功能测试 ===")
        
        # 测试获取机械臂位姿
        pos, euler = self.robot.get_xyz_eulerdeg()
        print(f"当前机械臂位置: {pos}")
        print(f"当前机械臂姿态: {euler}")
        
        # 测试计算相机位姿
        camera_pos, camera_euler = self.calculate_camera_pose_in_base()
        print(f"相机位置: {camera_pos}")
        print(f"相机姿态: {camera_euler}")
        
        # 测试像素坐标到基座坐标的转换
        print("\n--- 测试像素坐标转换 ---")
        test_distance = 300.0  # mm
        test_u = 640.0  # 图像中心附近的像素坐标
        test_v = 360.0
        
        print(f"测试参数: 距离={test_distance}mm, 像素坐标=({test_u}, {test_v})")
        test_object_pos = self.calculate_object_position_in_base(test_distance, test_u, test_v)
        print(f"计算得到的物体位置: {test_object_pos}")
        
        # 测试夹爪
        print("\n--- 测试夹爪开合 ---")
        self.gripper.open_gripper()
        time.sleep(1)
        self.gripper.close_gripper()
        time.sleep(1)
        self.gripper.open_gripper()
        
        print("系统测试完成")
    
    def test_pixel_to_base_conversion(self, test_cases):
        """
        测试多个像素坐标到基座坐标的转换
        
        Args:
            test_cases (list): 测试用例列表，每个元素为 (distance, u, v, description)
        """
        print("=== 批量测试像素坐标转换 ===")
        
        for i, (distance, u, v, description) in enumerate(test_cases):
            print(f"\n测试用例 {i+1}: {description}")
            print(f"输入参数: 距离={distance}mm, 像素坐标=({u}, {v})")
            
            try:
                object_pos = self.calculate_object_position_in_base(distance, u, v)
                print(f"计算结果: 基座坐标=[{object_pos[0]:.2f}, {object_pos[1]:.2f}, {object_pos[2]:.2f}] mm")
                
                # 检查坐标是否合理（简单的边界检查）
                if abs(object_pos[0]) > 1000 or abs(object_pos[1]) > 1000 or object_pos[2] < 0:
                    print("⚠️  警告: 计算出的坐标可能超出机械臂工作范围")
                else:
                    print("✅ 坐标在合理范围内")
                    
            except Exception as e:
                print(f"❌ 计算失败: {e}")
        
        print("\n批量测试完成")


# ============================================================================
# 简化的API接口，方便实际使用
# ============================================================================

def create_vision_pick_system(robot_ip='192.168.5.111', gripper_port='/dev/ttyUSB0'):
    """
    创建视觉抓取系统的简化接口
    
    Args:
        robot_ip (str): 机械臂IP地址
        gripper_port (str): 夹爪串口
    
    Returns:
        VisionBasedPickPlace: 视觉抓取系统实例
    """
    return VisionBasedPickPlace(robot_ip, gripper_port)

def execute_vision_pick(vision_system, distance_mm, pixel_u, pixel_v):
    """
    执行视觉抓取的简化接口
    
    Args:
        vision_system: 视觉抓取系统实例
        distance_mm (float): 距离(毫米)
        pixel_u (float): 像素坐标u
        pixel_v (float): 像素坐标v
    
    Returns:
        bool: 是否成功执行
    """
    try:
        vision_system.vision_based_pick_and_place(distance_mm, pixel_u, pixel_v)
        return True
    except Exception as e:
        print(f"视觉抓取失败: {e}")
        return False

def calculate_object_position(vision_system, distance_mm, pixel_u, pixel_v):
    """
    仅计算物体位置的简化接口
    
    Args:
        vision_system: 视觉抓取系统实例
        distance_mm (float): 距离(毫米)
        pixel_u (float): 像素坐标u
        pixel_v (float): 像素坐标v
    
    Returns:
        np.ndarray: 物体在基座系下的位置
    """
    return vision_system.calculate_object_position_in_base(distance_mm, pixel_u, pixel_v)

# 使用示例：
# system = create_vision_pick_system()
# success = execute_vision_pick(system, 300.0, 700.0, 300.0)
# position = calculate_object_position(system, 300.0, 700.0, 300.0)


def main():
    """
    主函数 - 演示如何使用视觉抓取系统
    """
    print("=== 视觉抓取系统演示程序 ===")
    
    # 创建视觉抓取系统实例
    vision_system = VisionBasedPickPlace()
    
    # 系统测试
    vision_system.test_system()
    
    # 批量测试不同像素坐标的转换效果
    print("\n=== 批量测试像素坐标转换 ===")
    test_cases = [
        (300, 637, 379, "图像中心位置"),
        (300, 700, 300, "图像右上方"),
        (300, 574, 379, "图像左侧"),
        (300, 637, 450, "图像下方"),
        (250, 800, 200, "近距离右上角"),
        (400, 500, 500, "远距离左下角")
    ]
    
    vision_system.test_pixel_to_base_conversion(test_cases)
    
    # 模拟接收到的距离信息和像素坐标(这里用固定值演示，实际应用中应该从视觉系统获取)
    print("\n--- 模拟视觉抓取任务 ---")
    
    # 示例1: 假设相机检测到物体距离为300mm，位于图像中心偏右上方
    distance_to_object = 300.0  # mm
    pixel_u = 700.0  # 水平像素坐标 (图像中心右侧)
    pixel_v = 300.0  # 垂直像素坐标 (图像中心上方)
    
    print(f"接收到视觉信息:")
    print(f"  - 距离: {distance_to_object}mm")
    print(f"  - 像素坐标: ({pixel_u}, {pixel_v})")
    
    # 执行基于视觉的抓取
    vision_system.vision_based_pick_and_place(distance_to_object, pixel_u, pixel_v)
    
    print("程序执行完成")


if __name__ == '__main__':
    main()
