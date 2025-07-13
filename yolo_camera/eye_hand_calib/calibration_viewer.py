import numpy as np


import sys
import os
import time
# 设置Qt插件路径，必须在导入PyQt5之前
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/home/adminpc/miniconda3/envs/camera/lib/python3.10/site-packages/PyQt5/Qt5/plugins'
os.environ['QT_PLUGIN_PATH'] = '/home/adminpc/miniconda3/envs/camera/lib/python3.10/site-packages/PyQt5/Qt5/plugins'
os.environ['QT_QPA_PLATFORM'] = 'xcb'

from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QLineEdit, QPushButton, QGridLayout, QGroupBox, QMessageBox, QDialog, QTabWidget
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox, QDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import yaml

# 添加模块路径
current_dir = os.path.dirname(os.path.abspath(__file__))
hardware_dir = os.path.join(current_dir, "..", "hardware")

# 添加各个硬件模块路径
robot_fr_dir = os.path.join(hardware_dir, "robot_fr", "fairino-python-sdk-master", "linux", "fairino")
robot_duco_dir = os.path.join(hardware_dir, "robot_duco")
grasper_dh_dir = os.path.join(hardware_dir, "grasper_dh", "GripperTestPython")
grasper_jodell_dir = os.path.join(hardware_dir, "grasper_jodell", "deepsense-jodell-python")
camera_util_dir = os.path.join(hardware_dir, "common_lib")
camera_realsense_dir = os.path.join(hardware_dir, "camera_realsenseD435")
camera_realsense_lib_dir = os.path.join(hardware_dir, "camera_realsenseD435", "lib")
camera_orbbec_dir = os.path.join(hardware_dir, "camera_orbbec")

# 添加gen_py模块路径 (Add gen_py module path)
gen_py_dir = "/home/adminpc/Documents/pick_place"

# 添加到sys.path
for path in [robot_fr_dir, robot_duco_dir, grasper_dh_dir, grasper_jodell_dir, 
             camera_util_dir, camera_realsense_dir, camera_realsense_lib_dir, camera_orbbec_dir, gen_py_dir]:
    abs_path = os.path.abspath(path)
    if abs_path not in sys.path:
        sys.path.append(abs_path)

print("已添加模块路径到sys.path")
from ViewCameraGroupBox import CameraGroupBox
from ViewJointGroupBox import JointGroupBox
from ViewTCPGroupBox import TCPGroupBox
from ViewGrasperGroupBox import GrasperGroupBox
from ViewTakePhotoGroupBox import TakePhotoGroupBox
from ViewCalibrationGroupBox import CalibrationGroupBox

import cv2

# 在导入cv2之后重新设置Qt插件路径
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/home/adminpc/miniconda3/envs/camera/lib/python3.10/site-packages/PyQt5/Qt5/plugins'
os.environ['QT_PLUGIN_PATH'] = '/home/adminpc/miniconda3/envs/camera/lib/python3.10/site-packages/PyQt5/Qt5/plugins'
os.environ['QT_QPA_PLATFORM'] = 'xcb'
class CameraWindow(QWidget):
    def __init__(self):
        super().__init__()

        # load config.yaml
        with open("config.yaml", "r") as f:
            self.config = yaml.safe_load(f)
            print("=====config begin=====")
            print(self.config)  
            print("=====config end=====")
            
        # config
        self.intrinsic_path = "data/intrinsic.txt"
        self.intrinsic = None
        self.poses_path = "data/poses.txt"

        self.setWindowTitle("RobotManager")
        self.setFixedSize(1080, 760) # (2240, 1400)

        # Create main layout
        main_layout = QVBoxLayout()

        # Create QTabWidget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Create Main Tab
        self.main_tab = QWidget()
        self.setup_main_tab()
        self.tab_widget.addTab(self.main_tab, "Main")

        # Create Demo Tab
        # self.demo_tab = DemoTab(self)
        # self.tab_widget.addTab(self.demo_tab, "Demo")

        self.setLayout(main_layout)

        

        # Initialize components (same as before)
        try:
            if self.config["camera"]["type"] == "orbbec":
                from orbbec_sdk_interface import OrbbecSDKInterface as CameraInterface
            elif self.config["camera"]["type"] == "realsense":
                from RealSenceInterface import RealSenseInterface as CameraInterface
            else:
                raise ValueError("Unsupported camera type in config.yaml")
            self.cam = CameraInterface()
            intrinsic = self.cam.get_camera_intrinsic()  # read as a numpy array
            print("intrinsic matrix:", intrinsic)  # save to a txt file with precision as 6
            np.savetxt("data/intrinsic.txt", intrinsic, fmt='%.6f')
        except:
            print("=====> Failed to initialize Camera Interface")
            self.cam = None

        try:
            if self.config['robot']['type'] == 'FAIR':
                import Robot
                self.robot = Robot.Robot(robot_ip=self.config['robot']['ip'])
            elif self.config['robot']['type'] == 'DUCO':
                from SiasunRobot import SiasunRobotPythonInterface as RobotInterface
            self.robot = RobotInterface(robot_ip=self.config['robot']['ip'])
        except Exception as e:
            print(f"Failed to initialize Robot Interface: {e}")
            self.robot = None

        try:
            if self.config['grasper']['type'] == 'DH':
                from DHGrasperInterface import DHGrasperInterface as GrasperInterface
                print("DHGrasperInterface")
                self.grasper = GrasperInterface(self.config['grasper']['port'])
                print("init")
            elif self.config['grasper']['type'] == 'EPG':
                from jodell_epg import EPG as GrasperInterface
                self.grasper = GrasperInterface(self.config['grasper']['port'])
                print("init")
                self.grasper.enable(True)
            # status_code, status_msg = self.grasper.get_status()
            # print(f"状态码: {status_code}")
            # print(f"状态信息: {status_msg}")
        except Exception as e:
            print(f"Failed to initialize Grasper Interface: {e}")
            self.grasper = None

        if self.cam is not None:
            self.intrinsic = self.cam.get_camera_intrinsic()
        else:
            self.intrinsic = np.loadtxt

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(50)  # Update every 50ms

        self.robot_timer = QTimer()
        self.robot_timer.timeout.connect(self.update_robot)
        self.robot_timer.start(1000)  # Update every 1 second

        self.grasper_timer = QTimer()
        self.grasper_timer.timeout.connect(self.update_grasper_status)
        self.grasper_timer.start(1000)  # Update every 1 second

    def setup_main_tab(self):
        """Setup the Main tab layout."""
        layout = QVBoxLayout()

        # Create top layout for image and depth labels
        top_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        middle_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        # Create a group box for camera controls
        self.camera_group_box = CameraGroupBox(self)
        left_layout.addWidget(self.camera_group_box)

        # Create a group box for joint controls
        self.joint_group_box = JointGroupBox(self)
        self.tcp_group_box = TCPGroupBox(self)
        middle_layout.addWidget(self.joint_group_box)
        middle_layout.addWidget(self.tcp_group_box)

        # Create a Grasper
        self.grasper_group_box = GrasperGroupBox(self)
        right_layout.addWidget(self.grasper_group_box)

        top_layout.addLayout(left_layout)
        top_layout.addLayout(middle_layout)
        top_layout.addLayout(right_layout)

        # Create a group box for calibration controls
        bottom_layout = QHBoxLayout()
        self.take_photo_box = TakePhotoGroupBox(self, "TakePhoto", self.poses_path)
        bottom_layout.addWidget(self.take_photo_box)
        self.calibration_group_box = CalibrationGroupBox(self)
        bottom_layout.addWidget(self.calibration_group_box)

        # Add top and bottom layouts to the main layout
        layout.addLayout(top_layout)
        layout.addLayout(bottom_layout)

        self.main_tab.setLayout(layout)

    def update_frame(self):
        if self.cam is None:
            return
        img, depth = self.cam.read_image_and_depth()
        self.camera_group_box.update_status(img, depth)
        # Display RGB image (assuming BGR format)
        

    def moveJ_adjust(self, label, delta_field, direction='+'):
        if self.robot is None:
            QMessageBox.warning(self, "Robot Not Connected", "The robot is not connected. Please check the connection and try again.")
            return
        # Extract joint index from the label text (e.g., "J1: 0.000")
        joint_index = int(label.text().split(":")[0][1:]) - 1  # Extract the number after 'J' and convert to 0-based index
        
        # Extract the adjustment value from the text field
        adjustment_text = delta_field.text()
        try:
            adjustment_val = float(adjustment_text)
        except ValueError:
            return

        # Determine the adjustment value
        adjustment = adjustment_val if direction == '+' else -adjustment_val
        
        # Get the current joint angles
        joints = self.robot.get_joint_degree()
        
        # Adjust the specific joint
        joints[joint_index] += adjustment
        
        # Move the robot to the new joint configuration
        self.robot.moveJ(joints)
        
        # Update the label to reflect the new joint angle
        label.setText(f"J{joint_index + 1}: {joints[joint_index]:.3f}")


    def moveJ_pose_adjust(self, label, delta_field, direction='+'):
        if self.robot is None:
            QMessageBox.warning(self, "Robot Not Connected", "The robot is not connected. Please check the connection and try again.")
            return

            
        # Extract position information from the label text (X, Y, Z, Rx, Ry, Rz)
        label_text = label.text().split(":")[0]
        if label_text in ["X", "Y", "Z"]:
            pose_index = 0
            pose_index = ["X", "Y", "Z"].index(label_text)
        else:
            pose_index = ["Rx", "Ry", "Rz"].index(label_text) + 3   
        trans, deg = self.robot.get_xyz_eulerdeg()
        # Extract the adjustment value from the text field
        adjustment_text = delta_field.text()
        try:
            adjustment_val = float(adjustment_text)
        except ValueError:
            return
        target_P = trans + deg
        # Determine the adjustment value
        target_P[pose_index] += adjustment_val if direction == '+' else -adjustment_val
        # Move the robot to the new joint configuration
        self.robot.moveJ_pose(target_P)
        # Update the label to reflect the new joint angle
        label.setText(f"{label_text}: {target_P[pose_index]:.3f}")
    
    def grasper_move_adjust(self, label, delta_field, direction='+'):
        # Extract position information from the label text (X, Y, Z, Rx, Ry, Rz)
        print("grasper_move_adjust called")   
        target_pos = self.grasper.get_position()
        print("current grasper:", target_pos)
        adjustment_text = delta_field.text()
        try:
            adjustment_val = int(adjustment_text)
        except ValueError:
            return
        
        target_pos += adjustment_val if direction == '+' else -adjustment_val
        # clip to 0, 255
        target_pos = max(0, min(self.config['grasper']['max_position'], target_pos))
        print("grasper:", target_pos)
        speed = 50  
        torque = 50
        self.grasper.move(position=target_pos, speed=speed, torque=torque)
        label.setText(f"Position: {target_pos}")

        
    def update_robot(self):
        if self.robot is None:
            return
        # Get the current joint angles from the robot
        joints = self.robot.get_joint_degree()
        trans, deg = self.robot.get_xyz_eulerdeg()

        self.joint_group_box.update_status(joints)
        self.tcp_group_box.update_status(trans, deg)
        
    def update_grasper_status(self):
        if self.grasper is None:
            return
        position = self.grasper.get_position()
        self.grasper_group_box.update_status(position)
    
    def detect_and_grasp_object(self, target_class="bottle"):
        """
        检测物体并获取抓取点的完整流程
        1. 读取并保存RGB和深度图像
        2. 使用YOLO检测指定类别的物体
        3. 计算物体中心点坐标和深度
        4. 保存结果到point.txt文件
        
        Args:
            target_class (str): 要检测的物体类别，默认为"bottle"
        """
        print("=== 开始物体检测和抓取点计算 ===")
        
        # 检查相机是否可用
        if self.cam is None:
            print("错误：相机未初始化或连接失败")
            return False
        
        try:
            # 1. 读取相机数据
            print("1. 正在读取相机数据...")
            rgb_image, depth_image = self.cam.read_image_and_depth()
            print(f"   RGB图像尺寸: {rgb_image.shape}")
            print(f"   深度图像尺寸: {depth_image.shape}")
            
            # 2. 创建保存目录
            save_dir = "/home/adminpc/Documents/复旦复合机器人/手眼标定demo/robotics_v20250704/robotics/eye_hand_calib/data/yolo_detections"
            os.makedirs(save_dir, exist_ok=True)
            print(f"2. 创建保存目录: {save_dir}")
            
            # 3. 保存RGB图像
            rgb_path = os.path.join(save_dir, 'rgb.png')
            # 注意：RealSense返回的是BGR格式，需要转换为RGB再保存
            rgb_to_save = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(rgb_path, cv2.cvtColor(rgb_to_save, cv2.COLOR_RGB2BGR))
            print(f"   RGB图像已保存到: {rgb_path}")
            
            # 4. 保存深度图像（以.npy格式保存原始数据）
            depth_npy_path = os.path.join(save_dir, 'depth.npy')
            np.save(depth_npy_path, depth_image)
            print(f"   深度数据已保存到: {depth_npy_path}")
            
            # 5. 保存深度可视化图像
            depth_vis_path = os.path.join(save_dir, 'depth_visualization.png')
            depth_vis = np.clip(depth_image, 0, 5000)  # 限制到5米
            depth_vis = (255 * depth_vis / 5000).astype(np.uint8)
            depth_vis_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            cv2.imwrite(depth_vis_path, depth_vis_color)
            print(f"   深度可视化图像已保存到: {depth_vis_path}")
            
            # 6. 使用YOLO检测物体
            print("3. 正在使用YOLO检测物体...")
            detections = self.yolo_detect_objects(rgb_to_save, target_class)
            
            if not detections:
                print(f"   未检测到目标物体: {target_class}")
                return False
            
            # 7. 选择置信度最高的检测结果
            best_detection = max(detections, key=lambda x: x['confidence'])
            print(f"   检测到 {len(detections)} 个 {target_class} 物体")
            print(f"   选择置信度最高的物体: {best_detection['confidence']:.3f}")
            
            # 8. 计算边界框中心点
            bbox = best_detection['bbox']  # [x1, y1, x2, y2]
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)
            print(f"4. 计算得到物体中心点: ({center_x}, {center_y})")
            
            # 9. 获取中心点的深度值
            center_depth = depth_image[center_y, center_x]
            print(f"5. 中心点深度值: {center_depth:.2f} mm")
            
            # 10. 在图像上绘制检测结果
            result_image = self.draw_detection_results(rgb_to_save.copy(), detections, (center_x, center_y))
            result_path = os.path.join(save_dir, 'detection_result.png')
            cv2.imwrite(result_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
            print(f"   检测结果图像已保存到: {result_path}")
            
            # 11. 保存抓取点信息到point.txt
            point_path = os.path.join(save_dir, 'point.txt')
            with open(point_path, 'w') as f:
                f.write(f"{center_x}\n")      # x坐标
                f.write(f"{center_y}\n")      # y坐标  
                f.write(f"{center_depth}\n")  # 深度值(mm)
            print(f"6. 抓取点信息已保存到: {point_path}")
            print(f"   格式: x坐标={center_x}, y坐标={center_y}, 深度={center_depth:.2f}mm")
            
            print("=== 物体检测和抓取点计算完成 ===")
            return True
            
        except Exception as e:
            print(f"错误：检测过程中发生异常: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def yolo_detect_objects(self, rgb_image, target_class="bottle"):
        """
        使用YOLO模型检测指定类别的物体
        
        Args:
            rgb_image (numpy.ndarray): RGB图像
            target_class (str): 目标类别名称
            
        Returns:
            list: 检测结果列表，每个元素包含bbox、confidence、class_name等信息
        """
        try:
            # 导入YOLO模型
            from ultralytics import YOLO
            
            # 加载预训练的YOLOv8模型
            model = YOLO('yolov8n.pt')  # 使用nano版本，速度较快
            print("   YOLO模型加载成功")
            
            # 运行检测
            results = model(rgb_image, verbose=False)
            
            # 处理检测结果
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        # 获取边界框坐标
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                        
                        # 获取置信度
                        confidence = boxes.conf[i].cpu().numpy()
                        
                        # 获取类别ID和名称
                        class_id = int(boxes.cls[i].cpu().numpy())
                        class_name = model.names[class_id].lower()
                        
                        # 只保留目标类别的检测结果
                        if class_name == target_class.lower() and confidence > 0.5:
                            detection = {
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(confidence),
                                'class_name': class_name,
                                'class_id': class_id
                            }
                            detections.append(detection)
                            print(f"   检测到 {class_name}: 置信度={confidence:.3f}, 位置=[{int(x1)},{int(y1)},{int(x2)},{int(y2)}]")
            
            return detections
            
        except ImportError:
            print("   错误：未安装ultralytics库，请运行: pip install ultralytics")
            return []
        except Exception as e:
            print(f"   YOLO检测错误: {e}")
            return []
    
    def draw_detection_results(self, image, detections, center_point):
        """
        在图像上绘制检测结果和中心点
        
        Args:
            image (numpy.ndarray): 输入图像
            detections (list): 检测结果列表
            center_point (tuple): 中心点坐标 (x, y)
            
        Returns:
            numpy.ndarray: 绘制了检测结果的图像
        """
        result_image = image.copy()
        
        # 绘制所有检测框
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # 绘制边界框
            cv2.rectangle(result_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # 绘制标签
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(result_image, (bbox[0], bbox[1] - label_size[1] - 10), 
                         (bbox[0] + label_size[0], bbox[1]), (0, 255, 0), -1)
            cv2.putText(result_image, label, (bbox[0], bbox[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # 绘制中心点
        center_x, center_y = center_point
        cv2.circle(result_image, (center_x, center_y), 8, (255, 0, 0), -1)  # 蓝色实心圆
        cv2.circle(result_image, (center_x, center_y), 15, (255, 0, 0), 2)   # 蓝色圆圈
        
        # 绘制十字准星
        cv2.line(result_image, (center_x - 20, center_y), (center_x + 20, center_y), (255, 0, 0), 2)
        cv2.line(result_image, (center_x, center_y - 20), (center_x, center_y + 20), (255, 0, 0), 2)
        
        # 添加中心点标签
        cv2.putText(result_image, f"Center: ({center_x},{center_y})", 
                   (center_x + 25, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        return result_image


if __name__ == '__main__':
    print("----------start")
    app = QApplication(sys.argv)
    
    # 模块导入已在文件开头处理
    try:
        import Robot
        from jodell_epg import EPG
        from demo_tab import DemoTab
        print("Required modules imported successfully")
    except Exception as e:
        print(f"Warning: Some modules failed to import: {e}")
    
    window = CameraWindow()
    window.show()
    print("-----------GUI launched")
    sys.exit(app.exec_())
