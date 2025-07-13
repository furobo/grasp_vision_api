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
from ViewTakePhotoGroupBox import TakePhotoGroupBox



class CalibrationGroupBox(QGroupBox):
    def __init__(self, parent=None, title="Calibration"):
        super().__init__(title)
        self.parent = parent
        self.intrinsic_path = parent.intrinsic_path
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # 显示相机内参
        self.intrinsic_label = QLabel("Intrinsic:\n(Not Loaded)")
        self.layout.addWidget(self.intrinsic_label)

        # 加载内参按钮
        load_intrinsic_button = QPushButton("Load Intrinsic")
        load_intrinsic_button.clicked.connect(self.load_intrinsic)
        self.layout.addWidget(load_intrinsic_button)

        # 显示标定结果
        self.result_label = QLabel("Calibration Result:\n(Not Calibrated)")
        self.layout.addWidget(self.result_label)

        # 触发标定按钮
        start_calib_button = QPushButton("Start Calibration")
        start_calib_button.clicked.connect(self.start_calibration)
        self.layout.addWidget(start_calib_button)
        self.load_intrinsic()

    def load_intrinsic(self):
        try:
            if self.parent.intrinsic_path and os.path.exists(self.parent.intrinsic_path):
                intrinsic = np.loadtxt(self.parent.intrinsic_path)
                self.parent.intrinsic = intrinsic
                intrinsic_text = "\n".join([", ".join([f"{val:.2f}" for val in row]) for row in intrinsic.reshape(3, 3)])
                self.intrinsic_label.setText(f"Intrinsic:\n{intrinsic_text}")
            else:
                self.intrinsic_label.setText("Intrinsic:\nFile not found.")
        except Exception as e:
            self.intrinsic_label.setText(f"Intrinsic:\nError loading file.\n{str(e)}")

    def start_calibration(self):
        import calibration_eyeonhand
        result = calibration_eyeonhand.start_calibration(intrinsic_path=self.intrinsic_path, num_photos=7)

        # Extract results
        mean_translation = result.get('mean_translation', [0, 0, 0])*1000
        rmse = result.get('rmse', 0)*1000
        mae = result.get('mae', 0)*1000

        # Update the calibration status label with the results
        individual_errors = result.get('individual_errors')*1000
        individual_errors_str = "\n".join([f"Error for pose {i}: {individual_errors[i]:.4f}" for i in range(len(individual_errors))])
        self.result_label.setText(
            f"Calibration Completed:\n"
            f"Mean Translation: {mean_translation[0]:.4f}, {mean_translation[1]:.4f}, {mean_translation[2]:.4f}\n"
            f"RMSE: {rmse:.4f}, MAE: {mae:.4f}\n{individual_errors_str}"
        )
