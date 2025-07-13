from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen, QImage
from PyQt5.QtCore import Qt
from PyQt5.QtCore import Qt, QPoint, QRect
from camera_util import read_exr_to_array
import cv2
import os
import numpy as np
from PyQt5.QtWidgets import (QScrollArea, QLabel, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QFrame, QWidget, QMainWindow, QAction, QRadioButton, QMessageBox, QTableWidget, QTableWidgetItem)
import open3d as o3d
import copy
import transforms3d as tfs
from pyvistaqt import QtInteractor
from PyQt5.QtWidgets import QSplitter
from world_image import WorldImage
from grasper_detector import GrasperDetector
import yaml

class DemoTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.manager = parent
        self.setup_ui()
        
        # 初始化抓取检测器
        self.load_config()
        self.grasp_detector = GraspDetector(
            yolo_model_path=self.config['yolo']['model_path'],
            conf_threshold=self.config['yolo']['conf_threshold']
        )
        
        # 设置世界图像实例
        self.world_image = WorldImage()
        self.grasp_detector.set_world_image(self.world_image)
    
    def load_config(self):
        """加载配置文件"""
        with open('config.yaml', 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
    
    def detect_grasp_targets(self):
        """检测抓取目标"""
        image_path = 'data/pose0/rgb.png'
        
        if not os.path.exists(image_path):
            print("图像文件不存在")
            return
        
        # 检测抓取目标
        target_classes = self.config['yolo']['target_classes']
        grasp_targets = self.grasp_detector.detect_grasp_targets(image_path, target_classes)
        
        print(f"检测到 {len(grasp_targets)} 个可抓取目标:")
        for target in grasp_targets:
            pos = target['base_position']
            print(f"  {target['class_name']} - 位置: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}] mm")
        
        # 可视化结果
        output_path = 'data/pose0/grasp_detection.png'
        self.grasp_detector.visualize_grasp_targets(image_path, grasp_targets, output_path)
        
        return grasp_targets
    


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

######################
# 2D图像显示界面
######################
class ImageLabel(QLabel):
    def __init__(self, parent=None, manager=None, status_label=None):
        super().__init__(parent)
        self.parent_tab = parent
        self.manager = manager
        self.intrinsic = self.manager.intrinsic
        self.tcp2base = get_tcp2base_matrix()
        self.eye2tcp = np.loadtxt('data/eye2tcp_matrix.txt') # mm
        self.status_label = status_label
        self.image_path = 'data/pose0/rgb.png'
        self.mask_path = 'data/pose0/mask.png'
        self.depth_path = 'data/pose0/depth.exr'
        self.local_mask_path = 'data/pose0/local_mask.png'
        self.camera_full_pcd_path = 'data/pose0/camera_full_pcd.ply' # camera frame
        self.base_full_pcd_path = 'data/pose0/base_full_pcd.ply'
        self.painter = QPainter(self)
        self.drawing_rect = False
        self.selection_start = None
        self.selection_end = None
        self.current_point = None
        self.show_mask = False
        self.show_local_mask = False
        self.setMouseTracking(True)
        self.update_image()
        self.save_full_ply()

    
    def paintEvent(self, event):
        super().paintEvent(event)
        self.painter.begin(self)
        self.painter.setPen(QPen(QColor('red'), 2))  # Set pen color and width
        if self.drawing_rect:
            rect = QRect(self.selection_start, self.selection_end)
            self.painter.drawRect(rect)  # Draw the rectangle
        self.painter.end()
    
    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            self.drawing_rect = True
            self.selection_start = event.pos()
            self.selection_end = event.pos()
            print("Left click on image label ")
            self.update()
        

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if event.button() == Qt.LeftButton:
            self.drawing_rect = False
            
        
    def mouseMoveEvent(self, event):
        print("image label mouse move event")
        super().mouseMoveEvent(event)
        self.current_point = event.pos()
        img_status = f"Current [Pixel: {self.current_point.x()}, {self.current_point.y()}]"
        base_pt = self.ws_image.get_point_position(self.current_point.y(), self.current_point.x(), frame_name='base')
        base_pt = [round(x, 1) for x in base_pt]
        img_status = img_status + f"    \t[Base Position: {base_pt[0], base_pt[1], base_pt[2]}]\n"
        if self.drawing_rect:
            self.selection_end = event.pos()
            img_status = img_status + f"Selection [Pixel: ({self.selection_start.x()}, {self.selection_start.y()}) to ({self.selection_end.x()}, {self.selection_end.y()})]"
            st_pt = self.ws_image.get_point_position(self.selection_start.y(), self.selection_start.x(), frame_name='base')
            ed_pt = self.ws_image.get_point_position(self.selection_end.y(), self.selection_end.x(), frame_name='base')
            st_pt = [round(x, 1) for x in st_pt]
            ed_pt = [round(x, 1) for x in ed_pt]
            img_status = img_status + f"    \t[Base Position: {st_pt[0], st_pt[1], st_pt[2]} to {ed_pt[0], ed_pt[1], ed_pt[2]}]"
            
        
        self.status_label.setText(img_status)
        self.update()
        
        
    def load_image(self):
        # read image as numpy array and convert to QImage
        self.ws_image = WorldImage()
        self.img = cv2.cvtColor(cv2.imread(self.image_path), cv2.COLOR_BGR2RGB)
        # self.img = cv2.imread(self.image_path)
        # 判断路径文件是否存在
        if os.path.exists(self.mask_path):
            mask = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)/255 # H,W
            mask = mask.astype(np.uint8)
            self.mask = np.stack([mask, np.zeros_like(mask), np.zeros_like(mask)], axis=-1)  
        else:
            self.mask = np.zeros_like(self.img)
        if os.path.exists(self.local_mask_path):
            local_mask = cv2.imread(self.local_mask_path, cv2.IMREAD_GRAYSCALE)/255 # H,W
            local_mask = local_mask.astype(np.uint8)
            self.local_mask = np.stack([local_mask, np.zeros_like(local_mask), np.zeros_like(local_mask)], axis=-1)  
        else:
            self.local_mask = np.zeros_like(self.img)
        
        self.depth = read_exr_to_array(self.depth_path)

    
    def update_image(self, show_mask=False, show_local_mask=False):
        """更新图像显示
        """
        self.load_image()
        if show_mask:
            img = cv2.addWeighted(self.img, 0.5, self.mask*255, 0.5, 0)  # Add mask to image with 50% transparency
        elif show_local_mask:
            img = cv2.addWeighted(self.img, 0.5, self.local_mask*255, 0.5, 0)  # Add local mask to image with 50% transparency
        else:
            img = self.img

        qimage = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        self.setPixmap(pixmap)
        self.update()
    
    def update_mask(self, mode):
        lefttop_x, lefttop_y = self.selection_start.x(), self.selection_start.y()
        rightbottom_x, rightbottom_y = self.selection_end.x(), self.selection_end.y()
        new_mask = np.zeros_like(self.mask[:, :, 0])
        new_mask[lefttop_y:rightbottom_y, lefttop_x:rightbottom_x] = 1
        new_mask = np.stack([new_mask, np.zeros_like(new_mask), np.zeros_like(new_mask)], axis=-1)  # Convert to 3-channel mask
        if mode == 'replace':
            self.mask = new_mask
        elif mode == 'overlay':
            self.mask[new_mask==1] = 1
        elif mode =='remove':
            self.mask[new_mask==1] = 0
        self.save_mask()
        self.ws_image.save_mask_ply()
        self.show_mask = True
        self.update_image(show_mask = self.show_mask)
        
    def save_mask(self):
        mask = self.mask[:, :, 0]*255
        cv2.imwrite(self.mask_path, mask)
        print(f"Mask saved to {self.mask_path}")
        
    def update_local_mask(self, mode):
        lefttop_x, lefttop_y = self.selection_start.x(), self.selection_start.y()
        rightbottom_x, rightbottom_y = self.selection_end.x(), self.selection_end.y()
        new_mask = np.zeros_like(self.mask[:, :, 0])
        new_mask[lefttop_y:rightbottom_y, lefttop_x:rightbottom_x] = 1
        new_mask = np.stack([new_mask, np.zeros_like(new_mask), np.zeros_like(new_mask)], axis=-1)  # Convert to 3-channel mask
        if mode == 'replace':
            self.local_mask = new_mask
        elif mode == 'overlay':
            self.local_mask[new_mask==1] = 1
        elif mode =='remove':
            self.local_mask[new_mask==1] = 0
        self.save_local_mask()
        self.ws_image.save_local_mask_ply()
        self.show_local_mask = True
        self.update_image(show_local_mask = self.show_local_mask)
        
        
    def save_local_mask(self):
        mask = self.local_mask[:, :, 0]*255
        cv2.imwrite(self.local_mask_path, mask)
        print(f"Local Mask saved to {self.local_mask_path}")
        
    def depth_to_pointcloud(self, mask=None):
        """深度图到点云
        """
        depth = copy.copy(self.depth)
        img = copy.copy(self.img)
        cam_K = self.intrinsic
        if mask is not None:
            depth[mask == 0] = 0
        vs, us = depth.nonzero()
        zs = depth[vs, us]
        xs = (us-cam_K[0, 2]) * zs / cam_K[0, 0]
        ys = (vs-cam_K[1, 2]) * zs / cam_K[1, 1]
        pts = np.stack([xs, ys, zs], axis=1)

        colors = img[vs, us] / 255.0
        return pts, colors, vs, us
    
    def depth_to_ply(self, ply_path, mask=None, frame_name='base'):
        """深度图到PLY文件
        """
        points, colors, _, _ = self.depth_to_pointcloud(mask)
        # 创建一个 Open3D 点云对象
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        if frame_name == 'base':
            print("tcp2base", self.tcp2base)
            print("eye2tcp", self.eye2tcp)
            transformation_matrix = self.tcp2base @ self.eye2tcp
            point_cloud.transform(transformation_matrix)
        o3d.io.write_point_cloud(ply_path, point_cloud)

    def save_full_ply(self):
        """保存完整点云
        """
        self.depth_to_ply(self.base_full_pcd_path, mask=None, frame_name='base')
        self.depth_to_ply(self.camera_full_pcd_path, mask=None, frame_name='camera')
    
    def save_mask_ply(self):
        """保存实例点云
        """
        mask = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)
        self.depth_to_ply(self.camera_mask_pcd_path, mask)
    
    
       
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
        
class DemoInteractor(QtInteractor):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.image_path = 'data/pose0/rgb.png'
        self.depth_path = 'data/pose0/depth.exr'
        self.camera_full_pcd_path = 'data/pose0/camera_full_pcd.ply' # camera frame
        self.base_full_pcd_path = 'data/pose0/base_full_pcd.ply'

        self.setMinimumWidth(1000)  # Set a fixed width for the plotter
        self.setMaximumWidth(1000)
        self.load_pointcloud()
        self.add_xyz_axis()

    def load_pointcloud(self):
        """Load and display a point cloud in the QtInteractor."""
        point_cloud = o3d.io.read_point_cloud(self.base_full_pcd_path)
        points = np.asarray(point_cloud.points)
        colors = np.asarray(point_cloud.colors)

        # Create a PyVista point cloud
        import pyvista as pv
        cloud = pv.PolyData(points)
        cloud['colors'] = (colors * 255).astype(np.uint8)

        # Add the point cloud to the QtInteractor
        self.add_points(cloud, scalars='colors', rgb=True)
        self.reset_camera()

    def add_xyz_axis(self):
        """Add XYZ axis to the plotter."""
        import pyvista as pv
        sphere = pv.Sphere(radius=10)
        self.add_mesh(sphere, show_edges=False, color='black') 
        start_point = np.array((0, 0, 0))  # Origin of the axes
        radius = 5
        height = 200
        for direction, color in zip(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),['red','green','blue']):
            line = pv.Cylinder(center=start_point+0.5*height*direction, direction=direction, radius=radius, height=height)
            self.add_mesh(line, color=color, opacity=0.5)
        
    def save_camera_pose(self):
        """save the camera pose in QtInteractor to camera_pose
        """
        self.camera_pose = None
    
    def restore_camera_pose(self):
        pass
        
        
        

class DemoTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.image_path = 'data/pose0/rgb.png'
        self.depth_path = 'data/pose0/depth.exr'
        self.setup_ui()

    def setup_ui(self):
        """Setup the layout and content for the Demo tab."""
        
        # Create a horizontal splitter to divide the image and point cloud display
        splitter = QSplitter(Qt.Horizontal)

        # Left side: Image display
        status_label = QLabel("status")
        image_label = ImageLabel(self, self.parent, status_label=status_label)
        splitter.addWidget(image_label)

        # Right side: Point cloud display using QtInteractor
        self.plotter = DemoInteractor(self)
        splitter.addWidget(self.plotter)

        # Set equal stretch factors for the splitter
        splitter.setStretchFactor(0, 1)  # Image label
        splitter.setStretchFactor(1, 1)  # Plotter

        # Set the splitter as the main layout
        layout = QVBoxLayout()
        layout.addWidget(splitter)
        layout.addWidget(status_label)
        self.setLayout(layout)

        
         # 添加YOLO检测按钮
        self.yolo_detect_btn = QPushButton("YOLO检测")
        self.yolo_detect_btn.clicked.connect(self.detect_grasp_targets)
