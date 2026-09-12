#!/usr/bin/env python3
"""
AVIS Engine <-> ROS 2 Bridge Node (Jazzy & Python 3.12 Compatible)

This node bridges the AVIS Engine simulator with ROS 2 autonomous driving pipelines.

Features:
  - Dynamic ROS 2 Parameters for IP, port, topic names, limits, and scaling.
  - Image publisher (sensor_msgs/Image) and CameraInfo publisher (sensor_msgs/CameraInfo).
  - Flexible command subscriber (/servo, cmd_vel) supporting std_msgs/Int8, std_msgs/Float32, and geometry_msgs/Twist.
  - Robust thread-safe socket management with automatic reconnection on simulator restart.
  - Comprehensive logging and graceful shutdown handling.

Usage:
  # Via ros2 run:
  ros2 run avis_bridge bridge_node

  # Or standalone script:
  python3 bridge_node.py
"""

import sys
import os
import threading
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rcl_interfaces.msg import SetParametersResult, ParameterType

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Int8, Float32
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

# Robust module imports supporting direct execution and package execution
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    from avisengine import Car
except ImportError:
    try:
        from avis_bridge.avisengine import Car
    except ImportError:
        from .avisengine import Car


class AvisBridgeNode(Node):
    """
    ROS 2 Bridge Node connecting AVIS Engine simulator with ROS 2 autonomy modules.
    """

    def __init__(self):
        super().__init__('avis_bridge_node')

        # ── 1. Declare ROS 2 Parameters ──────────────────────────────────────
        self.declare_parameter('simulator_ip', '127.0.0.1')
        self.declare_parameter('simulator_port', 25001)
        self.declare_parameter('image_topic', 'camera/image_raw')
        self.declare_parameter('camera_info_topic', 'camera/camera_info')
        self.declare_parameter('servo_topic', '/servo')
        self.declare_parameter('cmd_vel_topic', 'cmd_vel')
        self.declare_parameter('camera_frame_id', 'avis_camera')

        # Control scaling parameters
        self.declare_parameter('max_steering_avis', 100)
        self.declare_parameter('max_speed_avis', 350)
        self.declare_parameter('max_servo_input', 6)
        self.declare_parameter('max_cmd_vel_input', 6)
        self.declare_parameter('sensor_angle', 30)

        # Performance & Features
        self.declare_parameter('publish_camera_info', True)
        self.declare_parameter('reconnect_interval_sec', 2.0)

        # Retrieve initial parameter values
        self._ip = self.get_parameter('simulator_ip').value
        self._port = self.get_parameter('simulator_port').value
        self._image_topic = self.get_parameter('image_topic').value
        self._camera_info_topic = self.get_parameter('camera_info_topic').value
        self._servo_topic = self.get_parameter('servo_topic').value
        self._cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self._camera_frame_id = self.get_parameter('camera_frame_id').value

        self._max_steering_avis = self.get_parameter('max_steering_avis').value
        self._max_speed_avis = self.get_parameter('max_speed_avis').value
        self._max_servo_input = float(self.get_parameter('max_servo_input').value)
        self._max_cmd_vel_input = float(self.get_parameter('max_cmd_vel_input').value)
        self._sensor_angle = self.get_parameter('sensor_angle').value

        self._publish_camera_info = self.get_parameter('publish_camera_info').value
        self._reconnect_interval = self.get_parameter('reconnect_interval_sec').value

        # Register dynamic parameter update callback
        self.add_on_set_parameters_callback(self._on_parameter_change)

        # ── 2. Setup QoS Profiles ─────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        cmd_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # ── 3. Setup Publishers & Subscribers ─────────────────────────────────
        self.image_pub = self.create_publisher(Image, self._image_topic, sensor_qos)
        self.camera_info_pub = self.create_publisher(CameraInfo, self._camera_info_topic, sensor_qos)

        self.servo_sub = self.create_subscription(
            Int8, self._servo_topic, self._servo_callback, cmd_qos)
        self.cmd_vel_sub = self.create_subscription(
            Int8, self._cmd_vel_topic, self._cmd_vel_callback, cmd_qos)
        self.twist_sub = self.create_subscription(
            Twist, '/cmd_vel_twist', self._twist_callback, cmd_qos)

        self.bridge = CvBridge()

        # ── 4. Internal State & Thread Locks ─────────────────────────────────
        self._lock = threading.Lock()
        self._current_servo = 0.0      # Normalized or raw input units
        self._current_cmd_vel = 0.0    # Speed input units
        self._running = True

        # Initialize Car interface safely
        try:
            self.car = Car(server=self._ip, port=self._port)
        except TypeError:
            self.car = Car()

        # ── 5. Start Communication Worker Thread ──────────────────────────────
        self._thread = threading.Thread(target=self._communication_loop, daemon=True)
        self._thread.start()

        self.get_logger().info(
            f'AVIS Bridge Node initialized successfully.\n'
            f'  Server target: {self._ip}:{self._port}\n'
            f'  Image Pub: {self._image_topic}\n'
            f'  Servo Sub: {self._servo_topic} | CmdVel Sub: {self._cmd_vel_topic}'
        )

    # ── Dynamic Parameter Callback ────────────────────────────────────────────

    def _on_parameter_change(self, params):
        """Allows updating node parameters at runtime via ROS 2 tools."""
        for param in params:
            if param.name == 'simulator_ip':
                self._ip = param.value
            elif param.name == 'simulator_port':
                self._port = param.value
            elif param.name == 'max_steering_avis':
                self._max_steering_avis = param.value
            elif param.name == 'max_speed_avis':
                self._max_speed_avis = param.value
            elif param.name == 'max_servo_input':
                self._max_servo_input = float(param.value)
            elif param.name == 'max_cmd_vel_input':
                self._max_cmd_vel_input = float(param.value)
            elif param.name == 'sensor_angle':
                self._sensor_angle = param.value
            elif param.name == 'publish_camera_info':
                self._publish_camera_info = param.value
            elif param.name == 'reconnect_interval_sec':
                self._reconnect_interval = float(param.value)

        return SetParametersResult(successful=True)

    # ── Command Callbacks ─────────────────────────────────────────────────────

    def _servo_callback(self, msg: Int8):
        with self._lock:
            self._current_servo = float(msg.data)

    def _cmd_vel_callback(self, msg: Int8):
        with self._lock:
            self._current_cmd_vel = float(msg.data)

    def _twist_callback(self, msg: Twist):
        """Alternative subscriber for standard ROS 2 geometry_msgs/Twist commands."""
        with self._lock:
            self._current_cmd_vel = msg.linear.x
            self._current_servo = msg.angular.z

    # ── Camera Info Helper ────────────────────────────────────────────────────

    def _build_camera_info(self, stamp, width: int, height: int) -> CameraInfo:
        """Constructs camera calibration info for 3D vision and lane detection."""
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = self._camera_frame_id
        info.height = height
        info.width = width
        info.distortion_model = 'plumb_bob'

        # Pinhole camera intrinsic approximation for simulation camera
        fx = width * 0.8
        fy = height * 0.8
        cx = width / 2.0
        cy = height / 2.0

        info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        return info

    # ── Communication Loop ───────────────────────────────────────────────────

    def _communication_loop(self):
        """
        Background worker thread:
          1. Connects to Avis Engine simulator.
          2. Continuously sends command updates & receives image frames.
          3. Reconnects seamlessly if connection drops.
        """
        warmup_counter = 0
        warmup_frames = 3

        while self._running and rclpy.ok():
            # Connection establishment / retry logic
            if not self.car.is_connected:
                self.get_logger().warn(
                    f'Connecting to AVIS Engine at {self._ip}:{self._port} ...')
                if self.car.connect(self._ip, self._port):
                    self.get_logger().info('Successfully connected to AVIS Engine!')
                    warmup_counter = 0
                    time.sleep(1.0)
                else:
                    self.get_logger().error(
                        f'Connection failed. Retrying in {self._reconnect_interval}s...')
                    time.sleep(self._reconnect_interval)
                    continue

            # Thread-safe read of latest commands
            with self._lock:
                raw_servo = self._current_servo
                raw_cmd_vel = self._current_cmd_vel

            # Safely scale and clamp steering
            # Input [-max_servo_input, max_servo_input] -> Avis [-max_steering_avis, max_steering_avis]
            if self._max_servo_input > 0:
                normalized_servo = raw_servo / self._max_servo_input
            else:
                normalized_servo = 0.0
            normalized_servo = max(-1.0, min(1.0, normalized_servo))
            avis_steering = int(normalized_servo * self._max_steering_avis)

            # Safely scale and clamp speed
            # Input [-max_cmd_vel_input, max_cmd_vel_input] -> Avis [-max_speed_avis, max_speed_avis]
            if self._max_cmd_vel_input > 0:
                normalized_speed = raw_cmd_vel / self._max_cmd_vel_input
            else:
                normalized_speed = 0.0
            normalized_speed = max(-1.0, min(1.0, normalized_speed))
            avis_speed = int(normalized_speed * self._max_speed_avis)

            # Apply commands to car instance
            self.car.speed_value = avis_speed
            self.car.steering_value = avis_steering
            self.car.sensor_angle = self._sensor_angle

            # Fetch frame and sensor payload from simulator
            success = self.car.getData()
            if not success:
                self.get_logger().warn('Lost connection or failed to read data from AVIS Engine.')
                self.car.close()
                time.sleep(self._reconnect_interval)
                continue

            # Skip initial startup frames to prevent noise
            if warmup_counter < warmup_frames:
                warmup_counter += 1
                continue

            # Publish image and camera_info
            image = self.car.getImage()
            if image is not None and image.size > 0:
                stamp = self.get_clock().now().to_msg()

                # Convert OpenCV image to ROS 2 Image message
                img_msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
                img_msg.header.stamp = stamp
                img_msg.header.frame_id = self._camera_frame_id
                self.image_pub.publish(img_msg)

                # Publish CameraInfo
                if self._publish_camera_info:
                    h, w = image.shape[:2]
                    info_msg = self._build_camera_info(stamp, w, h)
                    self.camera_info_pub.publish(info_msg)

            time.sleep(0.001)

    # ── Node Shutdown ─────────────────────────────────────────────────────────

    def destroy_node(self):
        self.get_logger().info('Shutting down AVIS Bridge Node...')
        self._running = False
        if hasattr(self, 'car') and self.car:
            try:
                self.car.stop()
            except Exception:
                pass
        super().destroy_node()


# ── Entry Point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = AvisBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
