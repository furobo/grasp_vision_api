import numpy as np
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QLineEdit, QPushButton, QGridLayout, QGroupBox, QMessageBox, QDialog, QTabWidget
from PyQt5.QtCore import QTimer

import sys
import os
import time

from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox, QDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import yaml
from ViewCameraGroupBox import CameraGroupBox
from ViewJointGroupBox import JointGroupBox
from ViewTCPGroupBox import TCPGroupBox
from ViewGrasperGroupBox import GrasperGroupBox


class TakePhotoGroupBox(QGroupBox):
    def __init__(self, parent=None, title="TakePhoto", poses_path=None):
        super().__init__(title)
        self.parent = parent
        self.poses_path = poses_path
        

        # Create bottom layout for text fields and buttons
        if os.path.exists(self.poses_path):
            self.poses = np.loadtxt(self.poses_path, delimiter=",")
        else:
            self.poses = np.array([
                [45, -90, 30, -60, -90, 0],
                [45, -90, 30, -60, -90, 0],
                [45, -90, 30, -60, -90, 0],
                [45, -90, 30, -60, -90, 0],
                [45, -90, 30, -60, -90, 0],
                [45, -90, 30, -60, -90, 0],
                [45, -90, 30, -60, -90, 0],
            ])
            # save with precision 3 decimal places
            np.savetxt(self.poses_path, self.poses, delimiter=",", fmt="%.3f")

        self.text_fields = []
        layout = QGridLayout()

        for row in range(7):
            row_fields = []
            for col in range(6):
                text_field = QLineEdit()
                text_field.setText(f"{self.poses[row, col]:.3f}")
                text_field.textChanged.connect(lambda text, r=row, c=col: self.update_pose(r, c, text))
                layout.addWidget(text_field, row, col)
                row_fields.append(text_field)

            set_current_button = QPushButton("Set Current")
            set_current_button.clicked.connect(lambda _, r=row: self.set_current_pose(r))
            layout.addWidget(set_current_button, row, 6)

            take_photo_button = QPushButton("Take Photo")
            take_photo_button.clicked.connect(lambda _, r=row: self.take_photo(r))
            layout.addWidget(take_photo_button, row, 7)

            self.text_fields.append(row_fields)

        self.setLayout(layout)

    def update_pose(self, row, col, text):
        try:
            self.poses[row, col] = float(text)
        except ValueError:
            pass
    
    def set_current_pose(self, pose_index):
        joints = self.parent.robot.get_joint_degree()
        self.poses[pose_index] = joints
        for col in range(6):
            self.text_fields[pose_index][col].setText(f"{joints[col]:.3f}")
        np.savetxt(self.poses_path, self.poses, delimiter=",", fmt="%.3f")

    def get_text_fields(self):
        return self.text_fields
    

    def take_photo(self, pose_index=0):
        photo_dir = f"data/pose{pose_index}"
        if not os.path.exists(photo_dir):
            os.makedirs(photo_dir)

        pose = self.poses[pose_index]
        self.parent.robot.moveJ(pose)
        # wait a second for the robot to move
        time.sleep(1)
        trans, deg = self.parent.robot.get_xyz_eulerdeg()
        pose_path = os.path.join(photo_dir, "pose.txt")
        np.savetxt(pose_path, np.array(trans + deg), delimiter=",", fmt="%.6f")

        self.parent.cam.save_image_and_depth(photo_dir)