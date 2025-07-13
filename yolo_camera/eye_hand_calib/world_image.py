import numpy as np
import cv2
import os
import sys
import transforms3d as tfs

# 添加camera_util模块路径
current_dir = os.path.dirname(os.path.abspath(__file__))
camera_util_path = os.path.join(current_dir, "..", "hardware", "common_lib")
if camera_util_path not in sys.path:
    sys.path.append(camera_util_path)

from camera_util import read_exr_to_array
def get_tcp2base_matrix(tcp2base_path = "data/pose0/pose.txt", unit='mm'):
    """
    """
    tcp2base_pose = np.loadtxt(tcp2base_path) # mm, degree
    tcp2base_matrix = np.eye(4)
    if unit == 'm':
        tcp2base_matrix[:3, 3] = tcp2base_pose[:3]/1000 # convert mm to m
    else :
        tcp2base_matrix[:3, 3] = tcp2base_pose[:3]
    euler_angles = [c*np.pi/180 for c in tcp2base_pose[3:]] # convert degree to radian
    tcp2base_matrix[:3, :3] = tfs.euler.euler2mat(*euler_angles)
    return tcp2base_matrix

class WorldImage():
    """
    Class to handle the world image.
    """

    def __init__(self,  data_dir="data", pose_dir="data/pose0"):
        self.data_dir = data_dir
        self.pose_dir = pose_dir
        self.intrinsic = np.loadtxt(os.path.join(data_dir, 'intrinsic.txt'))
        self.eye2tcp = np.loadtxt(os.path.join(data_dir, 'eye2tcp_matrix.txt'))
        self.tcp2base = get_tcp2base_matrix(os.path.join(pose_dir, 'pose.txt'))
        self.image_path = os.path.join(pose_dir, 'rgb.png')
        self.depth_path = os.path.join(pose_dir, 'depth.exr')
        self.img = cv2.cvtColor(cv2.imread(self.image_path), cv2.COLOR_BGR2RGB)
        self.depth = read_exr_to_array(self.depth_path)

    def get_image(self, pose_index):
        pass


    def get_depth(self, pose_index):
        pass


    def get_point_cloud(self, pose_index, frame_name, mask):
        pass


    def get_pose(self, pose_index):
        pass


    def get_point_position(self, pixel_row, pixel_col, frame_name='eye'):
        """像素点转化相机或基坐标系下的点
        """
        z = self.depth[pixel_row, pixel_col]
        if z == 0:
            return np.array([0, 0, 0])
        x = (pixel_col-self.intrinsic[0, 2]) * z / self.intrinsic[0, 0]
        y = (pixel_row-self.intrinsic[1, 2]) * z / self.intrinsic[1, 1]
        if frame_name == 'eye':
            pt = np.array([x, y, z])
        elif frame_name == 'base':
            pt = self.tcp2base @ self.eye2tcp @ np.array([x, y, z, 1])
            pt = pt[:3]
        else:
            raise ValueError("Invalid frame name. Use 'eye' or 'base'.")
        return pt
        

if __name__ == '__main__':
    ws = WorldImage()
    r = ws.get_point_position(pixel_row=120, pixel_col=120, frame_name='base')
    print(r)
