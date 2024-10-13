# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ui_form.ui'
##
## Created by: Qt User Interface Compiler version 6.8.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QPushButton,
    QSizePolicy, QSlider, QSpacerItem, QVBoxLayout,
    QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1486, 618)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayout_2 = QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.frame = QFrame(self.centralwidget)
        self.frame.setObjectName(u"frame")
        self.frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame.setFrameShadow(QFrame.Shadow.Sunken)
        self.verticalLayout = QVBoxLayout(self.frame)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.label = QLabel(self.frame)
        self.label.setObjectName(u"label")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)

        self.verticalLayout.addWidget(self.label)

        self.LE_Power = QLineEdit(self.frame)
        self.LE_Power.setObjectName(u"LE_Power")
        self.LE_Power.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.LE_Power.setReadOnly(True)

        self.verticalLayout.addWidget(self.LE_Power)

        self.label_12 = QLabel(self.frame)
        self.label_12.setObjectName(u"label_12")
        sizePolicy.setHeightForWidth(self.label_12.sizePolicy().hasHeightForWidth())
        self.label_12.setSizePolicy(sizePolicy)

        self.verticalLayout.addWidget(self.label_12)

        self.LE_Servo = QLineEdit(self.frame)
        self.LE_Servo.setObjectName(u"LE_Servo")
        self.LE_Servo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.LE_Servo.setReadOnly(True)

        self.verticalLayout.addWidget(self.LE_Servo)

        self.label_2 = QLabel(self.frame)
        self.label_2.setObjectName(u"label_2")
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)

        self.verticalLayout.addWidget(self.label_2)

        self.LE_ControlManager = QLineEdit(self.frame)
        self.LE_ControlManager.setObjectName(u"LE_ControlManager")
        self.LE_ControlManager.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.LE_ControlManager.setReadOnly(True)

        self.verticalLayout.addWidget(self.LE_ControlManager)

        self.line = QFrame(self.frame)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.Shape.HLine)
        self.line.setFrameShadow(QFrame.Shadow.Sunken)

        self.verticalLayout.addWidget(self.line)

        self.PB_PowerOn = QPushButton(self.frame)
        self.PB_PowerOn.setObjectName(u"PB_PowerOn")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.PB_PowerOn.sizePolicy().hasHeightForWidth())
        self.PB_PowerOn.setSizePolicy(sizePolicy1)

        self.verticalLayout.addWidget(self.PB_PowerOn)

        self.PB_PowerOff = QPushButton(self.frame)
        self.PB_PowerOff.setObjectName(u"PB_PowerOff")
        sizePolicy1.setHeightForWidth(self.PB_PowerOff.sizePolicy().hasHeightForWidth())
        self.PB_PowerOff.setSizePolicy(sizePolicy1)

        self.verticalLayout.addWidget(self.PB_PowerOff)

        self.PB_ServoOn = QPushButton(self.frame)
        self.PB_ServoOn.setObjectName(u"PB_ServoOn")
        sizePolicy1.setHeightForWidth(self.PB_ServoOn.sizePolicy().hasHeightForWidth())
        self.PB_ServoOn.setSizePolicy(sizePolicy1)

        self.verticalLayout.addWidget(self.PB_ServoOn)

        self.PB_InitControlManager = QPushButton(self.frame)
        self.PB_InitControlManager.setObjectName(u"PB_InitControlManager")
        sizePolicy1.setHeightForWidth(self.PB_InitControlManager.sizePolicy().hasHeightForWidth())
        self.PB_InitControlManager.setSizePolicy(sizePolicy1)

        self.verticalLayout.addWidget(self.PB_InitControlManager)

        self.line_2 = QFrame(self.frame)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setFrameShape(QFrame.Shape.HLine)
        self.line_2.setFrameShadow(QFrame.Shadow.Sunken)

        self.verticalLayout.addWidget(self.line_2)

        self.PB_Zero = QPushButton(self.frame)
        self.PB_Zero.setObjectName(u"PB_Zero")
        sizePolicy1.setHeightForWidth(self.PB_Zero.sizePolicy().hasHeightForWidth())
        self.PB_Zero.setSizePolicy(sizePolicy1)

        self.verticalLayout.addWidget(self.PB_Zero)

        self.PB_Ready = QPushButton(self.frame)
        self.PB_Ready.setObjectName(u"PB_Ready")
        sizePolicy1.setHeightForWidth(self.PB_Ready.sizePolicy().hasHeightForWidth())
        self.PB_Ready.setSizePolicy(sizePolicy1)

        self.verticalLayout.addWidget(self.PB_Ready)

        self.PB_StartTeleoperation = QPushButton(self.frame)
        self.PB_StartTeleoperation.setObjectName(u"PB_StartTeleoperation")
        sizePolicy1.setHeightForWidth(self.PB_StartTeleoperation.sizePolicy().hasHeightForWidth())
        self.PB_StartTeleoperation.setSizePolicy(sizePolicy1)

        self.verticalLayout.addWidget(self.PB_StartTeleoperation)

        self.PB_StopMotion = QPushButton(self.frame)
        self.PB_StopMotion.setObjectName(u"PB_StopMotion")
        sizePolicy1.setHeightForWidth(self.PB_StopMotion.sizePolicy().hasHeightForWidth())
        self.PB_StopMotion.setSizePolicy(sizePolicy1)
        self.PB_StopMotion.setStyleSheet(u"QPushButton {background-color: #f16a6f}")

        self.verticalLayout.addWidget(self.PB_StopMotion)


        self.horizontalLayout_2.addWidget(self.frame)

        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.gridLayout_2 = QGridLayout()
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.label_3 = QLabel(self.centralwidget)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout_2.addWidget(self.label_3, 1, 0, 1, 1)

        self.frame_3 = QFrame(self.centralwidget)
        self.frame_3.setObjectName(u"frame_3")
        self.frame_3.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_3.setFrameShadow(QFrame.Shadow.Sunken)
        self.gridLayout_3 = QGridLayout(self.frame_3)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.LE_MasterArm = QLineEdit(self.frame_3)
        self.LE_MasterArm.setObjectName(u"LE_MasterArm")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.LE_MasterArm.sizePolicy().hasHeightForWidth())
        self.LE_MasterArm.setSizePolicy(sizePolicy2)
        self.LE_MasterArm.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.LE_MasterArm.setReadOnly(True)

        self.gridLayout_3.addWidget(self.LE_MasterArm, 0, 5, 1, 1)

        self.LE_Camera = QLineEdit(self.frame_3)
        self.LE_Camera.setObjectName(u"LE_Camera")
        sizePolicy2.setHeightForWidth(self.LE_Camera.sizePolicy().hasHeightForWidth())
        self.LE_Camera.setSizePolicy(sizePolicy2)
        self.LE_Camera.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.LE_Camera.setReadOnly(True)

        self.gridLayout_3.addWidget(self.LE_Camera, 0, 3, 1, 1)

        self.label_13 = QLabel(self.frame_3)
        self.label_13.setObjectName(u"label_13")
        font = QFont()
        font.setBold(True)
        self.label_13.setFont(font)
        self.label_13.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout_3.addWidget(self.label_13, 0, 0, 1, 1)

        self.label_15 = QLabel(self.frame_3)
        self.label_15.setObjectName(u"label_15")
        self.label_15.setFont(font)
        self.label_15.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout_3.addWidget(self.label_15, 0, 2, 1, 1)

        self.label_16 = QLabel(self.frame_3)
        self.label_16.setObjectName(u"label_16")
        self.label_16.setFont(font)
        self.label_16.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout_3.addWidget(self.label_16, 0, 4, 1, 1)

        self.LE_Robot = QLineEdit(self.frame_3)
        self.LE_Robot.setObjectName(u"LE_Robot")
        sizePolicy2.setHeightForWidth(self.LE_Robot.sizePolicy().hasHeightForWidth())
        self.LE_Robot.setSizePolicy(sizePolicy2)
        self.LE_Robot.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.LE_Robot.setReadOnly(True)

        self.gridLayout_3.addWidget(self.LE_Robot, 0, 1, 1, 1)


        self.gridLayout_2.addWidget(self.frame_3, 0, 0, 1, 3)

        self.lineEdit = QLineEdit(self.centralwidget)
        self.lineEdit.setObjectName(u"lineEdit")

        self.gridLayout_2.addWidget(self.lineEdit, 1, 1, 1, 2)

        self.lineEdit_2 = QLineEdit(self.centralwidget)
        self.lineEdit_2.setObjectName(u"lineEdit_2")

        self.gridLayout_2.addWidget(self.lineEdit_2, 2, 1, 1, 1)

        self.label_4 = QLabel(self.centralwidget)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout_2.addWidget(self.label_4, 2, 0, 1, 1)

        self.PB_Close = QPushButton(self.centralwidget)
        self.PB_Close.setObjectName(u"PB_Close")
        sizePolicy1.setHeightForWidth(self.PB_Close.sizePolicy().hasHeightForWidth())
        self.PB_Close.setSizePolicy(sizePolicy1)

        self.gridLayout_2.addWidget(self.PB_Close, 0, 3, 3, 1)

        self.pushButton_2 = QPushButton(self.centralwidget)
        self.pushButton_2.setObjectName(u"pushButton_2")

        self.gridLayout_2.addWidget(self.pushButton_2, 2, 2, 1, 1)

        self.gridLayout_2.setColumnStretch(0, 1)
        self.gridLayout_2.setColumnStretch(1, 2)
        self.gridLayout_2.setColumnStretch(2, 1)
        self.gridLayout_2.setColumnStretch(3, 1)

        self.verticalLayout_2.addLayout(self.gridLayout_2)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.PB_StartRecording = QPushButton(self.centralwidget)
        self.PB_StartRecording.setObjectName(u"PB_StartRecording")
        sizePolicy1.setHeightForWidth(self.PB_StartRecording.sizePolicy().hasHeightForWidth())
        self.PB_StartRecording.setSizePolicy(sizePolicy1)

        self.horizontalLayout.addWidget(self.PB_StartRecording)

        self.PB_StopRecording = QPushButton(self.centralwidget)
        self.PB_StopRecording.setObjectName(u"PB_StopRecording")
        sizePolicy1.setHeightForWidth(self.PB_StopRecording.sizePolicy().hasHeightForWidth())
        self.PB_StopRecording.setSizePolicy(sizePolicy1)
        self.PB_StopRecording.setStyleSheet(u"QPushButton {background-color: #f16a6f}")

        self.horizontalLayout.addWidget(self.PB_StopRecording)

        self.LE_Recording = QLineEdit(self.centralwidget)
        self.LE_Recording.setObjectName(u"LE_Recording")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.LE_Recording.sizePolicy().hasHeightForWidth())
        self.LE_Recording.setSizePolicy(sizePolicy3)
        self.LE_Recording.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.LE_Recording.setReadOnly(True)

        self.horizontalLayout.addWidget(self.LE_Recording)

        self.L_RecordingCount = QLabel(self.centralwidget)
        self.L_RecordingCount.setObjectName(u"L_RecordingCount")
        font1 = QFont()
        font1.setPointSize(13)
        self.L_RecordingCount.setFont(font1)
        self.L_RecordingCount.setFrameShape(QFrame.Shape.StyledPanel)
        self.L_RecordingCount.setFrameShadow(QFrame.Shadow.Sunken)
        self.L_RecordingCount.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout.addWidget(self.L_RecordingCount)

        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 1)
        self.horizontalLayout.setStretch(2, 1)
        self.horizontalLayout.setStretch(3, 1)

        self.verticalLayout_2.addLayout(self.horizontalLayout)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.frame_5 = QFrame(self.centralwidget)
        self.frame_5.setObjectName(u"frame_5")
        self.frame_5.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_5.setFrameShadow(QFrame.Shadow.Sunken)
        self.gridLayout_4 = QGridLayout(self.frame_5)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.label_14 = QLabel(self.frame_5)
        self.label_14.setObjectName(u"label_14")
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Preferred)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.label_14.sizePolicy().hasHeightForWidth())
        self.label_14.setSizePolicy(sizePolicy4)

        self.gridLayout_4.addWidget(self.label_14, 2, 0, 1, 1)

        self.PB_RotationZero = QPushButton(self.frame_5)
        self.PB_RotationZero.setObjectName(u"PB_RotationZero")

        self.gridLayout_4.addWidget(self.PB_RotationZero, 6, 1, 1, 1)

        self.label_18 = QLabel(self.frame_5)
        self.label_18.setObjectName(u"label_18")
        self.label_18.setFont(font)
        self.label_18.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout_4.addWidget(self.label_18, 0, 0, 1, 2)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.gridLayout_4.addItem(self.verticalSpacer, 4, 1, 1, 1)

        self.lineEdit_11 = QLineEdit(self.frame_5)
        self.lineEdit_11.setObjectName(u"lineEdit_11")
        self.lineEdit_11.setReadOnly(True)

        self.gridLayout_4.addWidget(self.lineEdit_11, 3, 1, 1, 1)

        self.S_Pitch = QSlider(self.frame_5)
        self.S_Pitch.setObjectName(u"S_Pitch")
        self.S_Pitch.setMinimum(-100)
        self.S_Pitch.setMaximum(100)
        self.S_Pitch.setOrientation(Qt.Orientation.Vertical)

        self.gridLayout_4.addWidget(self.S_Pitch, 6, 0, 1, 1)

        self.S_Yaw = QSlider(self.frame_5)
        self.S_Yaw.setObjectName(u"S_Yaw")
        self.S_Yaw.setMinimum(-100)
        self.S_Yaw.setMaximum(100)
        self.S_Yaw.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout_4.addWidget(self.S_Yaw, 5, 1, 1, 1)

        self.label_17 = QLabel(self.frame_5)
        self.label_17.setObjectName(u"label_17")

        self.gridLayout_4.addWidget(self.label_17, 3, 0, 1, 1)

        self.lineEdit_10 = QLineEdit(self.frame_5)
        self.lineEdit_10.setObjectName(u"lineEdit_10")
        self.lineEdit_10.setReadOnly(True)

        self.gridLayout_4.addWidget(self.lineEdit_10, 2, 1, 1, 1)

        self.line_3 = QFrame(self.frame_5)
        self.line_3.setObjectName(u"line_3")
        self.line_3.setFrameShape(QFrame.Shape.HLine)
        self.line_3.setFrameShadow(QFrame.Shadow.Sunken)

        self.gridLayout_4.addWidget(self.line_3, 1, 0, 1, 2)

        self.gridLayout_4.setColumnStretch(0, 1)
        self.gridLayout_4.setColumnStretch(1, 5)

        self.horizontalLayout_4.addWidget(self.frame_5)

        self.frame_2 = QFrame(self.centralwidget)
        self.frame_2.setObjectName(u"frame_2")
        self.frame_2.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_2.setFrameShadow(QFrame.Shadow.Sunken)
        self.verticalLayout_3 = QVBoxLayout(self.frame_2)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.label_7 = QLabel(self.frame_2)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout.addWidget(self.label_7, 0, 2, 1, 1)

        self.label_5 = QLabel(self.frame_2)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout.addWidget(self.label_5, 0, 0, 1, 1)

        self.label_6 = QLabel(self.frame_2)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout.addWidget(self.label_6, 0, 1, 1, 1)

        self.label_8 = QLabel(self.frame_2)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout.addWidget(self.label_8, 1, 0, 1, 1)

        self.label_9 = QLabel(self.frame_2)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout.addWidget(self.label_9, 1, 1, 1, 1)

        self.label_10 = QLabel(self.frame_2)
        self.label_10.setObjectName(u"label_10")
        self.label_10.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout.addWidget(self.label_10, 1, 2, 1, 1)


        self.verticalLayout_3.addLayout(self.gridLayout)


        self.horizontalLayout_4.addWidget(self.frame_2)

        self.frame_4 = QFrame(self.centralwidget)
        self.frame_4.setObjectName(u"frame_4")
        self.frame_4.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_4.setFrameShadow(QFrame.Shadow.Sunken)
        self.gridLayout_5 = QGridLayout(self.frame_4)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.line_4 = QFrame(self.frame_4)
        self.line_4.setObjectName(u"line_4")
        self.line_4.setFrameShape(QFrame.Shape.HLine)
        self.line_4.setFrameShadow(QFrame.Shadow.Sunken)

        self.gridLayout_5.addWidget(self.line_4, 1, 0, 1, 2)

        self.S_Y = QSlider(self.frame_4)
        self.S_Y.setObjectName(u"S_Y")
        self.S_Y.setMinimum(-100)
        self.S_Y.setMaximum(100)
        self.S_Y.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout_5.addWidget(self.S_Y, 5, 1, 1, 1)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.gridLayout_5.addItem(self.verticalSpacer_2, 4, 1, 1, 1)

        self.S_Z = QSlider(self.frame_4)
        self.S_Z.setObjectName(u"S_Z")
        self.S_Z.setMinimum(-100)
        self.S_Z.setMaximum(100)
        self.S_Z.setOrientation(Qt.Orientation.Vertical)

        self.gridLayout_5.addWidget(self.S_Z, 6, 0, 1, 1)

        self.label_20 = QLabel(self.frame_4)
        self.label_20.setObjectName(u"label_20")
        sizePolicy4.setHeightForWidth(self.label_20.sizePolicy().hasHeightForWidth())
        self.label_20.setSizePolicy(sizePolicy4)

        self.gridLayout_5.addWidget(self.label_20, 3, 0, 1, 1)

        self.label_19 = QLabel(self.frame_4)
        self.label_19.setObjectName(u"label_19")

        self.gridLayout_5.addWidget(self.label_19, 2, 0, 1, 1)

        self.label_21 = QLabel(self.frame_4)
        self.label_21.setObjectName(u"label_21")
        self.label_21.setFont(font)
        self.label_21.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout_5.addWidget(self.label_21, 0, 0, 1, 2)

        self.LE_Z = QLineEdit(self.frame_4)
        self.LE_Z.setObjectName(u"LE_Z")
        self.LE_Z.setReadOnly(True)

        self.gridLayout_5.addWidget(self.LE_Z, 3, 1, 1, 1)

        self.LE_Y = QLineEdit(self.frame_4)
        self.LE_Y.setObjectName(u"LE_Y")
        self.LE_Y.setReadOnly(True)

        self.gridLayout_5.addWidget(self.LE_Y, 2, 1, 1, 1)

        self.PB_PositionZero = QPushButton(self.frame_4)
        self.PB_PositionZero.setObjectName(u"PB_PositionZero")

        self.gridLayout_5.addWidget(self.PB_PositionZero, 6, 1, 1, 1)

        self.gridLayout_5.setColumnStretch(0, 1)
        self.gridLayout_5.setColumnStretch(1, 5)

        self.horizontalLayout_4.addWidget(self.frame_4)

        self.horizontalLayout_4.setStretch(0, 1)
        self.horizontalLayout_4.setStretch(1, 3)
        self.horizontalLayout_4.setStretch(2, 1)

        self.verticalLayout_2.addLayout(self.horizontalLayout_4)

        self.verticalLayout_2.setStretch(1, 1)
        self.verticalLayout_2.setStretch(2, 4)

        self.horizontalLayout_2.addLayout(self.verticalLayout_2)

        self.horizontalLayout_2.setStretch(0, 1)
        self.horizontalLayout_2.setStretch(1, 5)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Power", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"Servo", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Control Manager", None))
        self.PB_PowerOn.setText(QCoreApplication.translate("MainWindow", u"Power On", None))
        self.PB_PowerOff.setText(QCoreApplication.translate("MainWindow", u"Power Off", None))
        self.PB_ServoOn.setText(QCoreApplication.translate("MainWindow", u"Servo On", None))
        self.PB_InitControlManager.setText(QCoreApplication.translate("MainWindow", u"Initialize\n"
"Control Manager", None))
        self.PB_Zero.setText(QCoreApplication.translate("MainWindow", u"Zero", None))
        self.PB_Ready.setText(QCoreApplication.translate("MainWindow", u"Ready", None))
        self.PB_StartTeleoperation.setText(QCoreApplication.translate("MainWindow", u"Start\n"
"Teleoperation", None))
        self.PB_StopMotion.setText(QCoreApplication.translate("MainWindow", u"Stop", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Episode Name", None))
        self.label_13.setText(QCoreApplication.translate("MainWindow", u"Robot", None))
        self.label_15.setText(QCoreApplication.translate("MainWindow", u"Camera", None))
        self.label_16.setText(QCoreApplication.translate("MainWindow", u"MasterArm", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"# of ep.", None))
        self.PB_Close.setText(QCoreApplication.translate("MainWindow", u"Close", None))
        self.pushButton_2.setText(QCoreApplication.translate("MainWindow", u"Reset", None))
        self.PB_StartRecording.setText(QCoreApplication.translate("MainWindow", u"Start\n"
"Recording", None))
        self.PB_StopRecording.setText(QCoreApplication.translate("MainWindow", u"Stop\n"
"Recording", None))
        self.L_RecordingCount.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.label_14.setText(QCoreApplication.translate("MainWindow", u"Pitch", None))
        self.PB_RotationZero.setText(QCoreApplication.translate("MainWindow", u"Zero", None))
        self.label_18.setText(QCoreApplication.translate("MainWindow", u"\n"
"LEFT\n"
"", None))
        self.label_17.setText(QCoreApplication.translate("MainWindow", u"Yaw", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"Image Placeholder", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"Image Placeholder", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"Image Placeholder", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"Image Placeholder", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"Image Placeholder", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"Image Placeholder", None))
        self.label_20.setText(QCoreApplication.translate("MainWindow", u"Z", None))
        self.label_19.setText(QCoreApplication.translate("MainWindow", u"Y", None))
        self.label_21.setText(QCoreApplication.translate("MainWindow", u"\n"
"RIGHT\n"
"", None))
        self.PB_PositionZero.setText(QCoreApplication.translate("MainWindow", u"Zero", None))
    # retranslateUi

