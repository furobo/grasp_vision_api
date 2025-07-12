
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


if __name__ == "__main__": 
    # 创建实例
    robot = SiasunRobotPythonInterface() 
    
    # 获取当前位姿
    xyz, deg = robot.get_xyz_eulerdeg() 
    print(xyz, deg) 
    
    # 移动到目标位姿z轴增加20mm
    x,y,z = xyz
    robot.moveL([x,y,z-100], deg)



    




