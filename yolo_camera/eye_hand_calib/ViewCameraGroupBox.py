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


class CameraGroupBox(QGroupBox):
    def __init__(self, parent=None, title="Camera"):
        super().__init__(title)
        self.parent = parent

        self.max_depth = 3000

        self.image_label = QLabel("RGB Image")
        self.image_label.setFixedSize(300, 240)

        self.depth_label = QLabel("Depth Map")
        self.depth_label.setFixedSize(300, 240)

        self.explore_button = QPushButton("Explore")
        self.explore_button.clicked.connect(self.open_image_explorer)

        camera_group_layout = QVBoxLayout()
        image_layout = QHBoxLayout()
        image_layout.addWidget(self.image_label)
        image_layout.addWidget(self.explore_button)

        camera_group_layout.addLayout(image_layout)
        camera_group_layout.addWidget(self.depth_label)

        self.setLayout(camera_group_layout)
        self.setStyleSheet("QGroupBox { margin: 0px; }")

    def open_image_explorer(self, width=1280, height=720):
        """
        Opens a new window to display the RGB image
        """
        # create a image label from file path data/pose0/rgb.png
        

        dialog = QDialog(self)
        dialog.setWindowTitle("Image Explorer")
        dialog.setFixedSize(width, height)

        layout = QVBoxLayout()
        large_image_label = QLabel()
        large_image_label.setPixmap(self.image_label.pixmap().scaled(1200, 900, aspectRatioMode=Qt.KeepAspectRatio))
        layout.addWidget(large_image_label)

        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.close)
        layout.addWidget(close_button)

        dialog.setLayout(layout)
        dialog.exec_()

    def update_status(self, img, depth):
        rgb_image = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_BGR888)
        self.image_label.setPixmap(QPixmap.fromImage(rgb_image).scaled(self.image_label.size(), aspectRatioMode=1))

        # depth clip to max_depth
        depth = np.clip(depth, 0, self.max_depth)
        # Normalize and convert depth to 8-bit grayscale
        depth_normalized = (255 * depth / 3000).astype(np.uint8)
        depth_colored = np.stack([depth_normalized] * 3, axis=-1)  # convert to RGB-like for display
        depth_image = QImage(depth_colored.data, depth_colored.shape[1], depth_colored.shape[0], depth_colored.strides[0], QImage.Format_RGB888)
        self.depth_label.setPixmap(QPixmap.fromImage(depth_image).scaled(self.depth_label.size(), aspectRatioMode=1))
