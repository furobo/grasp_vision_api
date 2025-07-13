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
from ViewTCPGroupBox import TCPGroupBox
import yaml





class GrasperGroupBox(QGroupBox):
    def __init__(self, parent=None, title="Grasper Controls"):
        super().__init__(title)
        self.parent = parent
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        position_layout = QHBoxLayout()

        self.position_label = QLabel("Position")
        position_layout.addWidget(self.position_label)

        self.position_field = QLineEdit()
        self.position_field.setValidator(QtGui.QIntValidator(1, self.parent.config['grasper']['max_position']))
        self.position_field.setText("10")
        position_layout.addWidget(self.position_field)

        button_plus = QPushButton("+")
        button_plus.clicked.connect(lambda: self.parent.grasper_move_adjust(self.position_label, self.position_field, '+'))
        position_layout.addWidget(button_plus)

        button_minus = QPushButton("-")
        button_minus.clicked.connect(lambda: self.parent.grasper_move_adjust(self.position_label, self.position_field, '-'))
        position_layout.addWidget(button_minus)

        self.layout.addLayout(position_layout)

    def update_status(self, pos):
        self.position_label.setText(f"Position: {pos}")