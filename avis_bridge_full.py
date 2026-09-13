#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Int8, Float32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import socket
import base64
import numpy as np
import cv2
import threading
import time
import math


class AvisFullBridgeNode(Node):
    WHEELBASE = 0.15
    MAX_SPEED_MS = 0.8
    STEER_TO_RAD = math.radians(30.0) / 6.0

    def __init__(self):
        super().__init__('avis_full_bridge')

        self.host = '127.0.0.1'
        self.port = 25001
        self.camera_topic = '/camera/image_raw'

        actuator_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        rt_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.img_pub = self.create_publisher(Image, self.camera_topic, rt_qos)
        self.servo_sub = self.create_subscription(Int8, '/servo', self.servo_callback, actuator_qos)
        self.steer_cmd_sub = self.create_subscription(Float32, '/steering_cmd', self.steering_cmd_callback, actuator_qos)
        self.speed_sub = self.create_subscription(Int8, '/cmd_vel', self.speed_callback, actuator_qos)
        self.speed_feedback_pub = self.create_publisher(Float32, '/car/speed', rt_qos)
        self.odom_pub = self.create_publisher(Odometry, '/odom', odom_qos)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        self._publish_static_map_tf()

        self.sock = None
        self.running = True
        self.lock = threading.Lock()
        self.new_steer_event = threading.Event()

        self.current_speed_cmd = 60
        self.current_steer_cmd = 0
        self.frame_count = 0

        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_theta = 0.0
        self.last_odom_time = None
        self.sim_speed_ms = 0.0
        self.sim_steer_rad = 0.0

        self.socket_thread = threading.Thread(target=self._socket_loop, daemon=True)
        self.socket_thread.start()

        self.odom_timer = self.create_timer(1.0 / 30.0, self._publish_odom)
        self.get_logger().info(f"Bridge Target Socket: {self.host}:{self.port}")

    def steering_cmd_callback(self, msg: Float32):
        """Continuous high-resolution steering in [-1.0, 1.0] -> AvisEngine [-100, 100]."""
        with self.lock:
            self.has_continuous_steering = True
            val = float(np.clip(msg.data, -1.0, 1.0))
            self.current_steer_cmd = int(round(val * 100.0))
            self.sim_steer_rad = val * math.radians(30.0)
            self.new_steer_event.set()

    def servo_callback(self, msg: Int8):
        with self.lock:
            # Fallback for discrete servo messages ONLY if /steering_cmd is not in use
            if getattr(self, 'has_continuous_steering', False):
                return
            self.current_steer_cmd = int((msg.data / 6.0) * 100)
            self.sim_steer_rad = msg.data * self.STEER_TO_RAD
            self.new_steer_event.set()

    def speed_callback(self, msg: Int8):
        with self.lock:
            if msg.data > 0:
                # Map [1, 6] to AvisEngine throttle [35, 80] so the vehicle actively overcomes friction
                self.current_speed_cmd = int(30 + (msg.data / 6.0) * 50)
            elif msg.data < 0:
                self.current_speed_cmd = int((msg.data / 6.0) * 50)
            else:
                self.current_speed_cmd = 0
            self.sim_speed_ms = (msg.data / 6.0) * self.MAX_SPEED_MS

    def send_command(self, speed, steering):
        with self.lock:
            self.current_speed_cmd = int(speed)
            self.current_steer_cmd = int(steering)

    def _publish_odom(self):
        now = self.get_clock().now()
        if self.last_odom_time is None:
            self.last_odom_time = now
            return

        dt = (now - self.last_odom_time).nanoseconds / 1e9
        self.last_odom_time = now

        v = self.sim_speed_ms
        delta = self.sim_steer_rad

        if abs(delta) > 1e-6:
            R = self.WHEELBASE / math.tan(delta)
            dtheta = v * dt / R
            dx = R * (math.sin(self.odom_theta + dtheta) - math.sin(self.odom_theta))
            dy = R * (math.cos(self.odom_theta) - math.cos(self.odom_theta + dtheta))
        else:
            dx = v * math.cos(self.odom_theta) * dt
            dy = v * math.sin(self.odom_theta) * dt
            dtheta = 0.0

        self.odom_x += dx
        self.odom_y += dy
        self.odom_theta += dtheta

        odom_msg = Odometry()
        odom_msg.header.stamp = now.to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'
        odom_msg.pose.pose.position.x = self.odom_x
        odom_msg.pose.pose.position.y = self.odom_y
        odom_msg.pose.pose.position.z = 0.0
        qz = math.sin(self.odom_theta / 2.0)
        qw = math.cos(self.odom_theta / 2.0)
        odom_msg.pose.pose.orientation.x = 0.0
        odom_msg.pose.pose.orientation.y = 0.0
        odom_msg.pose.pose.orientation.z = qz
        odom_msg.pose.pose.orientation.w = qw
        odom_msg.twist.twist.linear.x = v
        odom_msg.twist.twist.angular.z = v * math.tan(delta) / self.WHEELBASE if abs(self.WHEELBASE) > 1e-6 else 0.0
        self.odom_pub.publish(odom_msg)

        t = TransformStamped()
        t.header.stamp = now.to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.odom_x
        t.transform.translation.y = self.odom_y
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(t)

    def _publish_static_map_tf(self):
        """Broadcasts static identity transform map -> odom so RViz2 default Fixed Frame (map) works out-of-the-box."""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.static_tf_broadcaster.sendTransform(t)

    def _socket_loop(self):
        while self.running and rclpy.ok():
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(10.0)
                self.sock.connect((self.host, self.port))
                self.get_logger().info("Connected to AVIS Engine simulator socket.")

                while self.running and rclpy.ok():
                    with self.lock:
                        spd = self.current_speed_cmd
                        steer = self.current_steer_cmd

                    # Send command: continuous throttle 60 keeps car engaged; speed limit is set in AvisEngine UI
                    cmd = f"Speed:{spd},Steering:{steer},ImageStatus:1,SensorStatus:0,GetSpeed:0,SensorAngle:30\n"
                    try:
                        self.sock.sendall(cmd.encode('utf-8'))
                    except Exception as e:
                        self.get_logger().warn(f"Socket send failed: {e}")
                        break

                    buffer = bytearray()
                    while self.running and rclpy.ok():
                        try:
                            chunk = self.sock.recv(131072)
                            if not chunk:
                                break
                            buffer.extend(chunk)
                            if b'<EOF>' in buffer:
                                break
                        except socket.timeout:
                            break
                        except Exception:
                            break

                    if not buffer or b'<EOF>' not in buffer:
                        continue

                    self.new_steer_event.clear()
                    self._process_rx_buffer(buffer)
                    self.frame_count += 1
                    if self.frame_count % 60 == 0:
                        self.get_logger().info(
                            f"[AVIS Bridge] Active: frame #{self.frame_count} | Steer={steer}"
                        )
                    # Zero-lag synchronization: wait briefly for the perception & control
                    # pipeline to process this frame and produce the fresh steering command (~3-4ms)
                    self.new_steer_event.wait(timeout=0.008)

            except Exception as e:
                time.sleep(2.0)
            finally:
                if self.sock:
                    try:
                        self.sock.close()
                    except Exception:
                        pass
                    self.sock = None

    def _process_rx_buffer(self, buffer: bytearray):
        # Direct byte parsing avoids expensive utf-8 string decoding on 200KB payload
        img_start = buffer.find(b'<image>')
        if img_start != -1:
            img_end = buffer.find(b'</image>', img_start)
            if img_end != -1:
                try:
                    img_bytes_b64 = buffer[img_start + 7:img_end]
                    img_bytes = base64.b64decode(img_bytes_b64)
                    np_arr = np.frombuffer(img_bytes, np.uint8)
                    cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if cv_img is not None:
                        img_msg = Image()
                        img_msg.header.stamp = self.get_clock().now().to_msg()
                        img_msg.header.frame_id = 'camera_link'
                        img_msg.height = cv_img.shape[0]
                        img_msg.width = cv_img.shape[1]
                        img_msg.encoding = 'bgr8'
                        img_msg.is_bigendian = 0
                        img_msg.step = cv_img.shape[1] * 3
                        img_msg.data = cv_img.tobytes()
                        self.img_pub.publish(img_msg)
                except Exception:
                    pass


def main(args=None):
    rclpy.init(args=args)
    node = AvisFullBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.running = False
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

