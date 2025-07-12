import math
import sys
import os
import transforms3d as tfs
# current_dir = os.path.dirname(os.path.abspath(__file__))
# add_lib_path = os.path.join(os.path.dirname(current_dir), 'common')
# print(add_lib_path)
# sys.path.append(current_dir)
# sys.path.append(add_lib_path)
import time
import numpy as np
from DucoCobot import DucoCobot

import dh_modbus_gripper
import time

'''
参数表
'''
start_x, start_y, start_z = 500, -300, 150
end_x, end_y, end_z = 500, 177, 200
# [376.3168454170227, -177.78697609901428, 349.3789732456207]
class SiasunRobotPythonInterface(object):
    def __init__(self, robot_ip='192.168.5.111'):  # 192.168.31.144 192.168.1.10 192.168.5.110 192.168.1.60
        super(SiasunRobotPythonInterface, self).__init__()
        self.robot_ip = robot_ip
        self.init_robot()
    

    def init_robot(self):
        self.duco = DucoCobot(self.robot_ip, 7003) # 7003 2000 2001 2011
        while self.duco.open() == -1:
            self.duco.close()
            time.sleep(1)
        self.duco.power_on(True)
        self.duco.enable(True)
        self.duco.stop(True)
        self.tcp_pose = None
        
    def get_RT_matrix(self):
        """获取旋转平移矩阵

        Returns:
            4*4 numpy array (float64): TCP2base的旋转平移矩阵
        """
        pose = self.duco.get_tcp_pose()
        trans = np.array(pose[:3]) * 1000
        deg = pose[3:]
        rotation_matrix = tfs.euler.euler2mat(deg[0],
                                              deg[1],
                                              deg[2])
        rt_matrix = np.eye(4)
        rt_matrix[:3, :3] = rotation_matrix
        rt_matrix[:3, 3] = trans
        return rt_matrix
    
    
    def get_xyz_eulerdeg(self):
        """获取欧拉角格式的位姿信息
        Returns:
            list: 位置信息(x,y,z) 
            list: 姿态信息(rx,ry,rz) 角度360度制
        """
        pose = self.duco.get_tcp_pose()
        trans = np.array(pose[:3]) * 1000
        deg = [c/math.pi* 180 for c in pose[3:]]
        
        return trans.tolist(), deg
    
    def moveJ(self, target_J, vel=30, acc=30):
        """调整关节到目标关节位置

        Args:
            target_J (list of degrees): 6个关节的目标位置

        Returns:
            list: 返回码列表([0] 成功 [errcode]错误码)
        """
        assert len(target_J)==6, 'The length of joint_list should be 6'
        # convert to radians
        target_J = np.radians(target_J).tolist()
        ret = self.duco.movej2(target_J, vel, acc, 0, True)
        return ret
    
    def moveJ_pose(self, target_P, vel=30, acc=30):
        """调整关节到目标关节位置
        Args:
            target_P : 目标点信息

        Returns:
            list: 返回码列表([0] 成功 [errcode]错误码)
        """
        trans_mm = np.array(target_P[0:3])/1000.0 # convert from millimeter to meter
        deg_radian = np.radians(target_P[3:6])
        target_P = trans_mm.tolist() + deg_radian.tolist()  # concat the two lists
        block = True
        print("moveJ_pose target_P (m, radians): ", target_P)
        ret = self.duco.movej_pose(target_P, vel, acc, 0, None, None, None, block)
        return ret
    
    def moveL(self, trans, deg, vel=0.4, acc=0.2):
        """直线运动

        Args:
            trans (list: [x,y,z]): 位姿信息 mm
            deg (list: [rx, ry, rz]): 角度信息 degree (360度)

        Returns:
            list: 返回码列表([0] 成功 [errcode]错误码)
        """
        trans_mm = np.array(trans)/1000.0 # convert from millimeter to meter
        deg_radian = np.radians(deg)
        target_P = trans_mm.tolist() + deg_radian.tolist()  # concat the two lists
        block = True
        print("moveL target_P (m, radians): ", target_P)
        # movel(self, p, v, a, r, q_near, tool, wobj, block, op=op_)
        ret = self.duco.movel(target_P, vel, acc, 0, None, None, None, block)
        return ret

        
    def moveArc(self, P1, P2, P3, vel=0.4, acc=0.2):
        """移动到起点并进行圆弧运动

        Args:
            P1 (list: [x, r, z, rx, ry, rz]): 起始点（mm, degree unit)
            P2 (list: [x, r, z, rx, ry, rz]): 中间点（mm, degree unit)
            P3 (list: [x, r, z, rx, ry, rz]): 结束点（mm, degree unit)

        Returns:
            list: 返回码列表([0] 成功 [errcode]错误码)
        """
        
        # move to start
        ret = self.moveL(P1[:3], P1[3:])
        print("start ret", ret)
        # if ret[0] != 0:
        #    print("MoveArc step MoveL error {}".format(ret[0]))
        #    return ret
        
        trans_mm = np.array(P2[:3])/1000.0 # convert from millimeter to meter
        deg_radian = np.radians(P2[3:])
        target_P2 = trans_mm.tolist() + deg_radian.tolist()  # concat the two lists
        trans_mm = np.array(P3[:3])/1000.0 # convert from millimeter to meter
        deg_radian = np.radians(P3[3:])
        target_P3 = trans_mm.tolist() + deg_radian.tolist()  # concat the two lists
        # movel(self, p, v, a, r, q_near, tool, wobj, block, op=op_)
        # movec(self, p1, p2, v, a, r, mode, q_near, tool, wobj, block, op=op_):
        # mode:姿态控制模式
        # 0：无约束姿态控制，姿态和轨迹无固定关系。
        # 1：有约束姿态控制，姿态和轨迹保持固定关系。
        mode = 0
        block = True
        ret = self.duco.movec(target_P2, target_P3, vel, acc, 0, mode, None, None, None, block)
        print("movec ret", ret)
        return ret
        

    
    def moveC(self, P2, P3, vel=100.0, acc=100.0):
        """圆弧运动

        Args:
            P2 (list: [x, r, z, rx, ry, rz]): 中间点（mm, degree unit)
            P3 (list: [x, r, z, rx, ry, rz]): 结束点（mm, degree unit)

        Returns:
            list: 返回码列表([0] 成功 [errcode]错误码)
        """    
        raise Exception("Not Implemented")

    # =============== 绝对位置控制增强方法 ===============
    
    def move_to_absolute_position(self, x, y, z, rx=None, ry=None, rz=None, 
                                motion_type='linear', vel=None, acc=None):
        """运动到指定的绝对位置（基于机器人底座坐标系）
        
        Args:
            x, y, z (float): 目标位置坐标 (mm)
            rx, ry, rz (float, optional): 目标姿态角度 (度). 如果为None，保持当前姿态
            motion_type (str): 运动类型 'linear'(直线) 或 'joint'(关节)
            vel (float, optional): 运动速度。linear模式单位m/s，joint模式单位度/s
            acc (float, optional): 运动加速度。linear模式单位m/s^2，joint模式单位度/s^2
            
        Returns:
            list: 返回码列表
        """
        # 获取当前姿态（如果没有指定目标姿态）
        if rx is None or ry is None or rz is None:
            _, current_orient = self.get_xyz_eulerdeg()
            rx = rx if rx is not None else current_orient[0]
            ry = ry if ry is not None else current_orient[1] 
            rz = rz if rz is not None else current_orient[2]
        
        target_pos = [x, y, z]
        target_orient = [rx, ry, rz]
        
        if motion_type.lower() == 'linear':
            vel = vel if vel is not None else 0.4  # 默认速度
            acc = acc if acc is not None else 0.2  # 默认加速度
            return self.moveL(target_pos, target_orient, vel, acc)
        elif motion_type.lower() == 'joint':
            vel = vel if vel is not None else 30   # 默认速度
            acc = acc if acc is not None else 30   # 默认加速度
            target_pose = target_pos + target_orient
            return self.moveJ_pose(target_pose, vel, acc)
        else:
            raise ValueError("motion_type must be 'linear' or 'joint'")
    
    def move_relative_to_base(self, dx, dy, dz, drx=0, dry=0, drz=0, 
                            motion_type='linear', vel=None, acc=None):
        """相对于机器人底座坐标系进行相对运动
        
        Args:
            dx, dy, dz (float): 位置增量 (mm)
            drx, dry, drz (float): 姿态增量 (度)
            motion_type (str): 运动类型 'linear'(直线) 或 'joint'(关节)
            vel, acc (float, optional): 运动速度和加速度
            
        Returns:
            list: 返回码列表
        """
        # 获取当前位置和姿态
        current_pos, current_orient = self.get_xyz_eulerdeg()
        
        # 计算目标位置
        target_x = current_pos[0] + dx
        target_y = current_pos[1] + dy  
        target_z = current_pos[2] + dz
        target_rx = current_orient[0] + drx
        target_ry = current_orient[1] + dry
        target_rz = current_orient[2] + drz
        
        print(f"相对运动: 当前位置 {current_pos} -> 目标位置 [{target_x}, {target_y}, {target_z}]")
        print(f"相对运动: 当前姿态 {current_orient} -> 目标姿态 [{target_rx}, {target_ry}, {target_rz}]")
        
        return self.move_to_absolute_position(target_x, target_y, target_z, 
                                            target_rx, target_ry, target_rz,
                                            motion_type, vel, acc)
    
    def execute_waypoints(self, waypoints, motion_type='linear', vel=None, acc=None, 
                         wait_time=1.0):
        """执行一系列绝对位置路径点
        
        Args:
            waypoints (list): 路径点列表。每个点可以是:
                             - [x, y, z] (保持当前姿态)
                             - [x, y, z, rx, ry, rz] (指定位置和姿态)
            motion_type (str): 运动类型 'linear' 或 'joint'
            vel, acc (float, optional): 运动参数
            wait_time (float): 每个点之间的等待时间(秒)
            
        Returns:
            list: 每个路径点的执行结果
        """
        results = []
        
        for i, waypoint in enumerate(waypoints):
            print(f"\n执行路径点 {i+1}/{len(waypoints)}: {waypoint}")
            
            if len(waypoint) == 3:  # 只有位置
                x, y, z = waypoint
                ret = self.move_to_absolute_position(x, y, z, motion_type=motion_type, 
                                                   vel=vel, acc=acc)
            elif len(waypoint) == 6:  # 位置+姿态
                x, y, z, rx, ry, rz = waypoint
                ret = self.move_to_absolute_position(x, y, z, rx, ry, rz, 
                                                   motion_type=motion_type, vel=vel, acc=acc)
            else:
                print(f"错误: 路径点 {i+1} 格式不正确，应为 [x,y,z] 或 [x,y,z,rx,ry,rz]")
                results.append([-1])
                continue
            
            results.append(ret)
            
            if ret[0] != 0:
                print(f"路径点 {i+1} 执行失败，错误码: {ret}")
                break
            else:
                print(f"路径点 {i+1} 执行成功")
                
            if i < len(waypoints) - 1:  # 不是最后一个点
                time.sleep(wait_time)
                
        return results
    
    def move_to_safe_position(self):
        """移动到安全位置（可根据实际情况修改坐标）
        
        Returns:
            list: 返回码
        """
        # 这里定义一个相对安全的位置，请根据实际机器人工作空间调整
        safe_pos = [376.30605697631836, -177.79159545898438, 349.36362504959106]  
        safe_orient = [-117.74253019596662, -5.130262234189182, -114.86968432916757]  # 使用初始化的欧拉角
        
        print("移动到安全位置...")
        return self.move_to_absolute_position(safe_pos[0], safe_pos[1], safe_pos[2],
                                            safe_orient[0], safe_orient[1], safe_orient[2],
                                            motion_type='joint')




class DHGrasperInterface:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.gripper = dh_modbus_gripper.dh_modbus_gripper()
        self.enable()

    def enable(self, flag=True):
        if flag:
            self.gripper.open(self.port, self.baudrate)
            self.gripper.Initialization()
            print("Send grip init...")

        # 等待初始化完成
        while self.gripper.GetInitState() != 1:
            time.sleep(0.2)

        print("Gripper initialized")

    def set_force(self, force):
        self.gripper.SetTargetForce(force)

    def set_speed(self, speed):
        self.gripper.SetTargetSpeed(speed)

    def move_to(self, position, block=True):
        self.gripper.SetTargetPosition(position)
        if block:
            while self.gripper.GetGripState() == 0:
                time.sleep(0.2)

    def open_gripper(self):
        self.move_to(1000)

    def close_gripper(self):
        self.move_to(0)

    def get_position(self):
        return self.gripper.GetCurrentPosition()

    def get_speed(self):
        return self.gripper.GetCurrentSpeed()

    def get_force(self):
        return self.gripper.GetCurrentForce()

    def get_state(self):
        return self.gripper.GetGripState()

    def close(self):
        self.gripper.close()
        
    def move(self, position, speed, torque):
        self.set_speed(speed)
        self.set_force(torque)
        self.move_to(position)


if __name__ == '__main__':
    robot = SiasunRobotPythonInterface()
    xyz, deg = robot.get_xyz_eulerdeg()
    print(xyz, deg)

    gripper = DHGrasperInterface('/dev/ttyUSB0', 115200)
    gripper.set_speed(50)
    gripper.set_force(50)
    
    # 初始化夹爪欧拉角
    print("初始化夹爪姿态")
    current_pos, current_euler = robot.get_xyz_eulerdeg()
    init_euler = [-90.01709163778153, 0.5653085040623331, -97.95937903864494]
    # print(current_pos)
    print(f"当前姿态: {current_euler}")
    print(f"目标姿态: {init_euler}")
    robot.move_to_absolute_position(current_pos[0], current_pos[1], current_pos[2], 
                                   init_euler[0], init_euler[1], init_euler[2], 
                                   motion_type='joint')
    print("夹爪姿态初始化完成")
    time.sleep(2)
        
    print("打开夹爪")
    gripper.open_gripper()
    time.sleep(1)

    print("开始移动机械臂到初位置")
    robot.move_to_absolute_position(start_x, start_y, start_z, 
                                   init_euler[0], init_euler[1], init_euler[2],
                                   motion_type = 'joint')
    time.sleep(1)
        
    print("关闭夹爪")
    gripper.close_gripper()
    time.sleep(1)

    print("开始移动机械臂到末位置")
    robot.move_to_absolute_position(end_x, end_y, end_z,
                                   init_euler[0], init_euler[1], init_euler[2], 
                                   motion_type = 'joint')
    time.sleep(1)

    print("再次打开夹爪")
    gripper.open_gripper()
    
    print("回到安全位置")
    
    robot.move_to_absolute_position(500, 177, 400,
                                   init_euler[0], init_euler[1], init_euler[2], 
                                   motion_type = 'joint')
    robot.move_to_safe_position()


