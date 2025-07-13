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
from ViewCameraGroupBox import CameraGroupBox



class JointGroupBox(QGroupBox):
    def __init__(self, parent, title="Joint Controls"):
        super().__init__(title)
        self.parent = parent
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.joint_labels = []

        for i in range(6):
            label = QLabel(f"J{i+1}: 0.000")
            label.setFixedWidth(120)
            self.joint_labels.append(label)

            joint_layout = QHBoxLayout()
            joint_layout.addWidget(label)

            delta_field = QLineEdit()
            delta_field.setText("1.000")
            delta_field.setFixedWidth(80)
            joint_layout.addWidget(delta_field)

            button_plus = QPushButton("+")
            button_plus.clicked.connect(lambda _, l=label, d=delta_field: self.parent.moveJ_adjust(l, d, '+'))
            joint_layout.addWidget(button_plus)

            button_minus = QPushButton("-")
            button_minus.clicked.connect(lambda _, l=label, d=delta_field: self.parent.moveJ_adjust(l, d, '-'))
            joint_layout.addWidget(button_minus)

            self.layout.addLayout(joint_layout)

    def update_status(self, joints):
        for i, joint in enumerate(joints):
            # Update the joint labels with the current joint angles
            self.joint_labels[i].setText(f"J{i+1}: {joint:.3f}")
        
        

    def get_labels(self):
        return self.joint_labels