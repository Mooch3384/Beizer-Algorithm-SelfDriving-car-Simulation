#!/usr/bin/env python3
import sys
import time
import json
import math
import threading
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Int8, Float32, String
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QHBoxLayout, QGridLayout, QLabel, QPlainTextEdit, QPushButton,
    QComboBox, QSpinBox, QGroupBox, QFrame
)
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont


class RosSignals(QObject):
    image_raw = pyqtSignal(object)
    filtered_debug = pyqtSignal(object)
    filtered_image = pyqtSignal(object)
    debug_image = pyqtSignal(object)
    steering_value = pyqtSignal(float)
    lane_status = pyqtSignal(str)
    servo = pyqtSignal(int)
    cmd_vel = pyqtSignal(int)
    odom = pyqtSignal(object)


class DashboardRosNode(Node):
    def __init__(self, signals: RosSignals):
        super().__init__('dashboard_gui_node')
        self.signals = signals
        self.br = CvBridge()

        rt_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(Image, '/camera/image_raw', self._on_image_raw, rt_qos)
        self.create_subscription(Image, '/filtered_debug', self._on_filtered_debug, rt_qos)
        self.create_subscription(Image, '/filtered_image', self._on_filtered_image, rt_qos)
        self.create_subscription(Image, '/debug_image', self._on_debug_image, rt_qos)
        self.create_subscription(Float32, '/steering_value', self._on_steering, rt_qos)
        self.create_subscription(String, '/lane_status', self._on_lane_status, rt_qos)
        self.create_subscription(Int8, '/servo', self._on_servo, rt_qos)
        self.create_subscription(Int8, '/cmd_vel', self._on_cmd_vel, rt_qos)
        self.create_subscription(Odometry, '/odom', self._on_odom, rt_qos)

        self.servo_pub = self.create_publisher(Int8, '/servo', QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=5))
        self.cmd_vel_pub = self.create_publisher(Int8, '/cmd_vel', QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=5))

    def _img_to_cv(self, msg):
        try:
            return self.br.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception:
            return None

    def _on_image_raw(self, msg):
        img = self._img_to_cv(msg)
        if img is not None:
            self.signals.image_raw.emit(img)

    def _on_filtered_debug(self, msg):
        img = self._img_to_cv(msg)
        if img is not None:
            self.signals.filtered_debug.emit(img)

    def _on_filtered_image(self, msg):
        img = self._img_to_cv(msg)
        if img is not None:
            self.signals.filtered_image.emit(img)

    def _on_debug_image(self, msg):
        img = self._img_to_cv(msg)
        if img is not None:
            self.signals.debug_image.emit(img)

    def _on_steering(self, msg):
        self.signals.steering_value.emit(float(msg.data))

    def _on_lane_status(self, msg):
        self.signals.lane_status.emit(msg.data)

    def _on_servo(self, msg):
        self.signals.servo.emit(int(msg.data))

    def _on_cmd_vel(self, msg):
        self.signals.cmd_vel.emit(int(msg.data))

    def _on_odom(self, msg):
        self.signals.odom.emit(msg)

    def publish_emergency_stop(self):
        s = Int8()
        s.data = 0
        self.servo_pub.publish(s)
        c = Int8()
        c.data = 0
        self.cmd_vel_pub.publish(c)

    def publish_speed_override(self, val):
        c = Int8()
        c.data = int(max(-6, min(6, val)))
        self.cmd_vel_pub.publish(c)


class RosWorker(QThread):
    def __init__(self, signals: RosSignals):
        super().__init__()
        self.signals = signals
        self.node = None

    def run(self):
        rclpy.init()
        self.node = DashboardRosNode(self.signals)
        try:
            rclpy.spin(self.node)
        except Exception:
            pass

    def stop(self):
        if self.node:
            self.node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


class ImagePanel(QWidget):
    def __init__(self, title):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("color: #00ff00; font-weight: bold; font-size: 12px;")
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(320, 240)
        self.image_label.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333;")
        self.fps_label = QLabel("-- FPS | --x--")
        self.fps_label.setAlignment(Qt.AlignCenter)
        self.fps_label.setStyleSheet("color: #aaa; font-size: 10px;")
        layout.addWidget(self.title_label)
        layout.addWidget(self.image_label, stretch=1)
        layout.addWidget(self.fps_label)
        self._frame_times = deque(maxlen=30)
        self._last_shape = None

    def update_image(self, cv_img):
        now = time.time()
        self._frame_times.append(now)
        if len(self._frame_times) >= 2:
            dt = self._frame_times[-1] - self._frame_times[0]
            fps = (len(self._frame_times) - 1) / dt if dt > 0 else 0
        else:
            fps = 0
        h, w = cv_img.shape[:2]
        self._last_shape = (w, h)
        if len(cv_img.shape) == 2:
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        elif cv_img.shape[2] == 4:
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        max_w = self.image_label.width()
        max_h = self.image_label.height()
        if max_w > 0 and max_h > 0:
            scale = min(max_w / w, max_h / h)
            nw, nh = int(w * scale), int(h * scale)
            rgb = cv2.resize(rgb, (nw, nh))
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))
        self.fps_label.setText(f"{fps:.1f} FPS | {w}x{h}")


class VisionTab(QWidget):
    def __init__(self):
        super().__init__()
        grid = QGridLayout(self)
        grid.setSpacing(4)
        self.panels = [
            ImagePanel("Raw Simulator (/camera/image_raw)"),
            ImagePanel("BEV Segmented (/filtered_debug)"),
            ImagePanel("Binary Lane Mask (/filtered_image)"),
            ImagePanel("Lane HUD (/debug_image)"),
        ]
        for i, p in enumerate(self.panels):
            grid.addWidget(p, i // 2, i % 2)

    def on_image_raw(self, img):
        self.panels[0].update_image(img)

    def on_filtered_debug(self, img):
        self.panels[1].update_image(img)

    def on_filtered_image(self, img):
        self.panels[2].update_image(img)

    def on_debug_image(self, img):
        self.panels[3].update_image(img)


class TopicEchoTab(QWidget):
    TOPICS = ['/steering_value', '/lane_status', '/odom', '/servo', '/cmd_vel']

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        self.selector = QComboBox()
        self.selector.addItems(self.TOPICS)
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setCheckable(True)
        self.clear_btn = QPushButton("Clear")
        top.addWidget(QLabel("Topic:"))
        top.addWidget(self.selector)
        top.addWidget(self.pause_btn)
        top.addWidget(self.clear_btn)
        layout.addLayout(top)
        self.viewer = QPlainTextEdit()
        self.viewer.setReadOnly(True)
        self.viewer.setMaximumBlockCount(500)
        self.viewer.setFont(QFont("Monospace", 9))
        self.viewer.setStyleSheet(
            "background-color: #0d0d0d; color: #00ff00; border: 1px solid #333; padding: 4px;"
        )
        layout.addWidget(self.viewer)
        self.clear_btn.clicked.connect(self.viewer.clear)
        self._paused = False
        self.pause_btn.toggled.connect(lambda v: setattr(self, '_paused', v))
        self._buffers = {t: deque(maxlen=200) for t in self.TOPICS}
        self._latest = {}

    def push_data(self, topic, text):
        self._buffers[topic].append(text)
        self._latest[topic] = text
        if not self._paused and self.selector.currentText() == topic:
            self.viewer.appendPlainText(text)

    def on_steering(self, val):
        self.push_data('/steering_value', f"[{time.strftime('%H:%M:%S')}] steering={val:+.4f}")

    def on_lane_status(self, data):
        try:
            d = json.loads(data)
            line = f"[{time.strftime('%H:%M:%S')}] mode={d.get('mode','?')} detected={d.get('detected')} curv={d.get('curvature',-1):.1f}"
        except Exception:
            line = f"[{time.strftime('%H:%M:%S')}] {data}"
        self.push_data('/lane_status', line)

    def on_servo(self, val):
        self.push_data('/servo', f"[{time.strftime('%H:%M:%S')}] servo={val}")

    def on_cmd_vel(self, val):
        self.push_data('/cmd_vel', f"[{time.strftime('%H:%M:%S')}] cmd_vel={val}")

    def on_odom(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        theta = math.atan2(2.0 * (ori.w * ori.z + ori.x * ori.y),
                           1.0 - 2.0 * (ori.y * ori.y + ori.z * ori.z))
        v = msg.twist.twist.linear.x
        line = f"[{time.strftime('%H:%M:%S')}] x={pos.x:.3f} y={pos.y:.3f} θ={math.degrees(theta):.1f}° v={v:.3f}m/s"
        self.push_data('/odom', line)


class GaugeBar(QFrame):
    def __init__(self, label, min_val, max_val, parent=None):
        super().__init__(parent)
        self.min_val = min_val
        self.max_val = max_val
        self.current_val = 0
        self.label_text = label
        self.setFixedSize(200, 40)
        self.setStyleSheet("background: #1a1a1a; border: 1px solid #444; border-radius: 4px;")

    def set_value(self, val):
        self.current_val = max(self.min_val, min(self.max_val, val))
        self.update()

    def paintEvent(self, event):
        from PyQt5.QtGui import QPainter, QColor
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor("#1a1a1a"))
        range_val = self.max_val - self.min_val
        if range_val == 0:
            return
        ratio = (self.current_val - self.min_val) / range_val
        bar_w = int(ratio * (w - 4))
        center_x = int((0 - self.min_val) / range_val * (w - 4)) + 2
        if self.current_val >= 0:
            painter.fillRect(center_x, 4, bar_w - center_x + 2, h - 8, QColor("#00cc44"))
        else:
            neg_w = center_x - (int(ratio * (w - 4)) + 2)
            painter.fillRect(int(ratio * (w - 4)) + 2, 4, neg_w, h - 8, QColor("#cc4400"))
        painter.setPen(QColor("#ffffff"))
        painter.setFont(QFont("Monospace", 8))
        painter.drawText(4, h - 4, f"{self.label_text}: {self.current_val}")


class TelemetryTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        act_group = QGroupBox("Actuation")
        act_layout = QHBoxLayout(act_group)
        self.servo_gauge = GaugeBar("Servo", -6, 6)
        self.speed_gauge = GaugeBar("Speed", -6, 6)
        self.steer_cont_gauge = GaugeBar("Steer", -100, 100)
        act_layout.addWidget(self.servo_gauge)
        act_layout.addWidget(self.speed_gauge)
        act_layout.addWidget(self.steer_cont_gauge)
        act_layout.addStretch()
        layout.addWidget(act_group)

        status_group = QGroupBox("Tracking Status")
        status_layout = QHBoxLayout(status_group)
        self.mode_badge = QLabel("VISION_TRACKING")
        self.mode_badge.setFixedSize(160, 30)
        self.mode_badge.setAlignment(Qt.AlignCenter)
        self.mode_badge.setStyleSheet("background: #00cc44; color: white; font-weight: bold; border-radius: 4px;")
        self.curv_label = QLabel("Curvature: INF")
        self.curv_label.setStyleSheet("color: #00ff00; font-size: 14px; font-weight: bold;")
        self.detected_label = QLabel("Detected: True")
        self.detected_label.setStyleSheet("color: #00ff00; font-size: 14px;")
        status_layout.addWidget(self.mode_badge)
        status_layout.addWidget(self.curv_label)
        status_layout.addWidget(self.detected_label)
        status_layout.addStretch()
        layout.addWidget(status_group)

        odom_group = QGroupBox("Odometry")
        odom_layout = QGridLayout(odom_group)
        self.odom_x = QLabel("X: 0.000 m")
        self.odom_y = QLabel("Y: 0.000 m")
        self.odom_theta = QLabel("θ: 0.0°")
        self.odom_v = QLabel("v: 0.000 m/s")
        for lbl in [self.odom_x, self.odom_y, self.odom_theta, self.odom_v]:
            lbl.setStyleSheet("color: #00ff00; font-size: 14px; font-family: Monospace;")
        odom_layout.addWidget(self.odom_x, 0, 0)
        odom_layout.addWidget(self.odom_y, 0, 1)
        odom_layout.addWidget(self.odom_theta, 1, 0)
        odom_layout.addWidget(self.odom_v, 1, 1)
        layout.addWidget(odom_group)
        layout.addStretch()

    def on_servo(self, val):
        self.servo_gauge.set_value(val)

    def on_cmd_vel(self, val):
        self.speed_gauge.set_value(val)

    def on_steering(self, val):
        self.steer_cont_gauge.set_value(int(val * 100))

    def on_lane_status(self, data):
        try:
            d = json.loads(data)
            mode = d.get("mode", "UNKNOWN")
            curv = d.get("curvature", -1.0)
            det = d.get("detected", False)
            self.mode_badge.setText(mode)
            if mode == "DEAD_RECKONING":
                self.mode_badge.setStyleSheet("background: #cc4400; color: white; font-weight: bold; border-radius: 4px;")
            else:
                self.mode_badge.setStyleSheet("background: #00cc44; color: white; font-weight: bold; border-radius: 4px;")
            curv_str = f"{curv:.0f}px" if curv > 0 else "INF"
            self.curv_label.setText(f"Curvature: {curv_str}")
            self.detected_label.setText(f"Detected: {det}")
            self.detected_label.setStyleSheet(
                "color: #00ff00; font-size: 14px;" if det else "color: #cc4400; font-size: 14px;"
            )
        except Exception:
            pass

    def on_odom(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        theta = math.atan2(2.0 * (ori.w * ori.z + ori.x * ori.y),
                           1.0 - 2.0 * (ori.y * ori.y + ori.z * ori.z))
        v = msg.twist.twist.linear.x
        self.odom_x.setText(f"X: {pos.x:.3f} m")
        self.odom_y.setText(f"Y: {pos.y:.3f} m")
        self.odom_theta.setText(f"θ: {math.degrees(theta):.1f}°")
        self.odom_v.setText(f"v: {v:.3f} m/s")


class ControlTab(QWidget):
    def __init__(self, ros_worker: RosWorker):
        super().__init__()
        self.ros_worker = ros_worker
        layout = QVBoxLayout(self)

        estop_group = QGroupBox("Emergency Controls")
        estop_layout = QVBoxLayout(estop_group)
        self.estop_btn = QPushButton("EMERGENCY STOP")
        self.estop_btn.setFixedHeight(60)
        self.estop_btn.setStyleSheet(
            "background-color: #cc0000; color: white; font-size: 18px; font-weight: bold; border-radius: 8px;"
        )
        self.estop_btn.clicked.connect(self._emergency_stop)
        estop_layout.addWidget(self.estop_btn)
        layout.addWidget(estop_group)

        speed_group = QGroupBox("Manual Speed Override")
        speed_layout = QHBoxLayout(speed_group)
        self.speed_spin = QSpinBox()
        self.speed_spin.setRange(-6, 6)
        self.speed_spin.setValue(0)
        self.send_speed_btn = QPushButton("Send Speed")
        self.send_speed_btn.clicked.connect(self._send_speed)
        speed_layout.addWidget(QLabel("Speed [-6, 6]:"))
        speed_layout.addWidget(self.speed_spin)
        speed_layout.addWidget(self.send_speed_btn)
        speed_layout.addStretch()
        layout.addWidget(speed_group)
        layout.addStretch()

    def _emergency_stop(self):
        if self.ros_worker.node:
            self.ros_worker.node.publish_emergency_stop()

    def _send_speed(self):
        if self.ros_worker.node:
            self.ros_worker.node.publish_speed_override(self.speed_spin.value())


class DashboardGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AVIS Autonomous Driving Dashboard")
        self.setMinimumSize(1100, 700)
        self.setStyleSheet("background-color: #121212; color: #e0e0e0;")

        self.signals = RosSignals()
        self.ros_worker = RosWorker(self.signals)

        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #333; background: #1a1a1a; }
            QTabBar::tab { background: #2a2a2a; color: #ccc; padding: 8px 16px; margin: 2px; }
            QTabBar::tab:selected { background: #006600; color: white; }
        """)

        self.vision_tab = VisionTab()
        self.echo_tab = TopicEchoTab()
        self.telemetry_tab = TelemetryTab()
        self.control_tab = ControlTab(self.ros_worker)

        tabs.addTab(self.vision_tab, "Vision Grid")
        tabs.addTab(self.echo_tab, "Topic Echo")
        tabs.addTab(self.telemetry_tab, "Telemetry")
        tabs.addTab(self.control_tab, "Control Panel")

        self.setCentralWidget(tabs)

        self.signals.image_raw.connect(self.vision_tab.on_image_raw)
        self.signals.filtered_debug.connect(self.vision_tab.on_filtered_debug)
        self.signals.filtered_image.connect(self.vision_tab.on_filtered_image)
        self.signals.debug_image.connect(self.vision_tab.on_debug_image)

        self.signals.steering_value.connect(self.echo_tab.on_steering)
        self.signals.lane_status.connect(self.echo_tab.on_lane_status)
        self.signals.servo.connect(self.echo_tab.on_servo)
        self.signals.cmd_vel.connect(self.echo_tab.on_cmd_vel)
        self.signals.odom.connect(self.echo_tab.on_odom)

        self.signals.steering_value.connect(self.telemetry_tab.on_steering)
        self.signals.lane_status.connect(self.telemetry_tab.on_lane_status)
        self.signals.servo.connect(self.telemetry_tab.on_servo)
        self.signals.cmd_vel.connect(self.telemetry_tab.on_cmd_vel)
        self.signals.odom.connect(self.telemetry_tab.on_odom)

        self.ros_worker.start()

    def closeEvent(self, event):
        self.ros_worker.stop()
        self.ros_worker.wait(3000)
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = DashboardGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

