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
from ViewJointGroupBox import JointGroupBox
import yaml



class TCPGroupBox(QGroupBox):
    def __init__(self, parent, title="TCP Controls"):
        super().__init__(title)
        self.parent = parent
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.tcp_labels = []

        for name in ["X", "Y", "Z", "Rx", "Ry", "Rz"]:
            label = QLabel(f"{name}: 0.000")
            label.setFixedWidth(120)
            self.tcp_labels.append(label)

            tcp_layout = QHBoxLayout()
            tcp_layout.addWidget(label)

            delta_field = QLineEdit()
            delta_field.setText("10.000")
            delta_field.setFixedWidth(80)
            tcp_layout.addWidget(delta_field)

            button_plus = QPushButton("+")
            button_plus.clicked.connect(lambda _, l=label, d=delta_field: self.parent.moveJ_pose_adjust(l, d, '+'))
            tcp_layout.addWidget(button_plus)

            button_minus = QPushButton("-")
            button_minus.clicked.connect(lambda _, l=label, d=delta_field: self.parent.moveJ_pose_adjust(l, d, '-'))
            tcp_layout.addWidget(button_minus)

            self.layout.addLayout(tcp_layout)
    
    def update_status(self, trans, deg):
        self.tcp_labels[0].setText(f"X: {trans[0]:.3f}")
        self.tcp_labels[1].setText(f"Y: {trans[1]:.3f}")
        self.tcp_labels[2].setText(f"Z: {trans[2]:.3f}")
        self.tcp_labels[3].setText(f"Rx: {deg[0]:.3f}")
        self.tcp_labels[4].setText(f"Ry: {deg[1]:.3f}")
        self.tcp_labels[5].setText(f"Rz: {deg[2]:.3f}")

    def get_labels(self):
        return self.tcp_labels