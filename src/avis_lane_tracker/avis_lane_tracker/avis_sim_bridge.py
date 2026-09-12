import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Quaternion
from cv_bridge import CvBridge
import socket
import sys
import math
import time
import base64
import re
import numpy as np
import cv2
import threading


class AvisSimBridgeNode(Node):
    """
    ROS 2 Bridge Node for AvisEngine Simulation.
    Connects to AvisEngine TCP socket (localhost:25001), decodes camera frames and vehicle telemetry,
    and publishes standard ROS 2 messages to /camera/image_raw and /odom.
    """

    def __init__(self):
        super().__init__('avis_sim_bridge')

        # Parameters
        self.declare_parameter('host', '127.0.0.1')
        self.declare_parameter('port', 25001)
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('frame_rate', 30.0)

        self.host = self.get_parameter('host').get_parameter_value().string_value
        self.port = self.get_parameter('port').get_parameter_value().integer_value
        self.camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.frame_rate = self.get_parameter('frame_rate').get_parameter_value().double_value

        # Publishers
        self.img_pub = self.create_publisher(Image, self.camera_topic, 10)
        self.odom_pub = self.create_publisher(Odometry, self.odom_topic, 10)

        self.bridge = CvBridge()
        self.running = True
        self.sock = None

        # Simulated state if socket is pending connection
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.speed = 2.0  # m/s default simulation speed
        self.last_time = self.get_clock().now()

        # Start socket thread
        self.socket_thread = threading.Thread(target=self._socket_loop, daemon=True)
        self.socket_thread.start()

        # Timer for fallback/synthetic telemetry stream if simulator socket is standalone
        self.timer = self.create_timer(1.0 / self.frame_rate, self._timer_callback)

        self.get_logger().info(f"AvisEngine Bridge initialized. Target socket: {self.host}:{self.port}")

    def _socket_loop(self):
        """Background thread loop for TCP socket connection with AvisEngine."""
        while self.running and rclpy.ok():
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(2.0)
                self.sock.connect((self.host, self.port))
                self.get_logger().info(f"Connected to AvisEngine simulator at {self.host}:{self.port}")

                buffer = ""
                while self.running and rclpy.ok():
                    data = self.sock.recv(4096)
                    if not data:
                        break
                    buffer += data.decode('utf-8', errors='ignore')

                    # Parse XML/socket messages from AvisEngine
                    while '</EOF>' in buffer or '\n' in buffer:
                        if '</EOF>' in buffer:
                            parts = buffer.split('</EOF>', 1)
                            msg = parts[0]
                            buffer = parts[1]
                        else:
                            parts = buffer.split('\n', 1)
                            msg = parts[0]
                            buffer = parts[1]

                        self._process_message(msg)

            except (socket.timeout, ConnectionRefusedError, OSError):
                # Socket not available yet; sleep before retrying
                time.sleep(2.0)
            except Exception as e:
                self.get_logger().warn(f"Socket connection error: {e}")
                time.sleep(2.0)
            finally:
                if self.sock:
                    try:
                        self.sock.close()
                    except Exception:
                        pass

    def _process_message(self, msg: str):
        """Parse AvisEngine XML/socket message payload."""
        now = self.get_clock().now().to_msg()

        # Check for image tag
        if '<image>' in msg and '</image>' in msg:
            img_str = msg.split('<image>')[1].split('</image>')[0]
            try:
                img_bytes = base64.b64decode(img_str)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if cv_img is not None:
                    img_msg = self.bridge.cv2_to_imgmsg(cv_img, encoding='bgr8')
                    img_msg.header.stamp = now
                    img_msg.header.frame_id = 'camera_link'
                    self.img_pub.publish(img_msg)
            except Exception as e:
                self.get_logger().error(f"Failed to decode socket camera frame: {e}")

        # Check for speed/telemetry tag
        if '<speed>' in msg and '</speed>' in msg:
            try:
                speed_str = msg.split('<speed>')[1].split('</speed>')[0]
                self.speed = float(speed_str) / 3.6  # Convert km/h to m/s
            except ValueError:
                pass

    def _timer_callback(self):
        """
        Publish odometry message. If no socket data is available, compute synthetic motion
        forward to maintain a continuous, valid odometry stream for local mapping.
        """
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now

        if dt <= 0.0 or dt > 1.0:
            dt = 1.0 / self.frame_rate

        # Simple kinematic model for forward driven trajectory
        self.x += self.speed * math.cos(self.yaw) * dt
        self.y += self.speed * math.sin(self.yaw) * dt

        # Publish nav_msgs/msg/Odometry
        odom_msg = Odometry()
        odom_msg.header.stamp = now.to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'

        odom_msg.pose.pose.position.x = self.x
        odom_msg.pose.pose.position.y = self.y
        odom_msg.pose.pose.position.z = 0.0

        # Yaw to quaternion
        half_yaw = self.yaw * 0.5
        odom_msg.pose.pose.orientation.z = math.sin(half_yaw)
        odom_msg.pose.pose.orientation.w = math.cos(half_yaw)

        odom_msg.twist.twist.linear.x = self.speed
        odom_msg.twist.twist.angular.z = 0.0

        self.odom_pub.publish(odom_msg)

    def destroy_node(self):
        self.running = False
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AvisSimBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
