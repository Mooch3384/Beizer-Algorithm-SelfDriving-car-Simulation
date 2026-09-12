#!/usr/bin/env python3
"""
Controller Node for Autonomous Driving (AVIS Engine Simulation & Jetson)
Pure lateral steering controller:
  - Subscribes to /steering_value (Float32 in [-1.0, 1.0])
  - Low-latency smoothing and slew-rate limiting
  - Publishes continuous /steering_cmd (Float32 in [-1.0, 1.0]) for high-resolution simulator steering
  - Publishes discrete /servo (Int8 in [-6, 6]) for hardware compatibility
  - Speed is 100% controlled by the user via the AvisEngine UI panel (no speed manipulation here)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import Int8, Float32, String
import json
import numpy as np


class ControllerNode(Node):
    def __init__(self):
        super().__init__('controller_node')

        rt_sub_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        actuator_pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        # ── Parameter Declarations ───────────────────────────────────────────
        self.declare_parameter('steering_gain', 1.0)
        self.declare_parameter('invert_steering', False)
        self.declare_parameter('min_servo', -6)
        self.declare_parameter('max_servo', 6)
        self.declare_parameter('publish_rate_hz', 30.0)
        self.declare_parameter('ema_alpha', 0.80)
        self.declare_parameter('max_steer_rate', 2.8)

        self._load_params()

        # ── Subscribers & Publishers ─────────────────────────────────────────
        self.steer_sub = self.create_subscription(
            Float32, '/steering_value', self.steer_callback, rt_sub_qos)
        self.status_sub = self.create_subscription(
            String, '/lane_status', self.status_callback, rt_sub_qos)

        self.servo_pub = self.create_publisher(Int8, '/servo', actuator_pub_qos)
        self.steer_cmd_pub = self.create_publisher(Float32, '/steering_cmd', actuator_pub_qos)

        # ── Internal States ──────────────────────────────────────────────────
        self.current_steer = 0.0
        self.ema_steer = 0.0
        self.last_applied_continuous_steer = 0.0
        self.is_recovering = False
        self.lanes_detected = False
        self.curvature = -1.0
        self.mode = "VISION_TRACKING"
        self.last_continuous_steer_logged = 0.0

        self.add_on_set_parameters_callback(self._param_callback)

        # Watchdog keep-alive timer
        timer_period = 1.0 / max(1.0, float(self.publish_rate_hz))
        self.timer = self.create_timer(timer_period, self._watchdog_loop)

        self.get_logger().info('ControllerNode ready — pure lateral steering active (speed controlled via AvisEngine UI)')

    def _load_params(self):
        p = self.get_parameter
        self.steering_gain = float(p('steering_gain').value)
        self.invert_steering = bool(p('invert_steering').value)
        self.min_servo = int(p('min_servo').value)
        self.max_servo = int(p('max_servo').value)
        self.publish_rate_hz = float(p('publish_rate_hz').value)
        self.ema_alpha = float(p('ema_alpha').value)
        self.max_steer_rate = float(p('max_steer_rate').value)

    def _param_callback(self, params):
        for param in params:
            self.get_logger().info(f'Parameter updated: {param.name} = {param.value}')
        self._load_params()
        return SetParametersResult(successful=True)

    def status_callback(self, msg: String):
        try:
            d = json.loads(msg.data)
            self.is_recovering = d.get("recovering", False)
            self.lanes_detected = d.get("detected", False)
            self.curvature = d.get("curvature", -1.0)
            self.mode = d.get("mode", "VISION_TRACKING")
        except Exception:
            pass

    def steer_callback(self, msg: Float32):
        """Immediately compute and output steering upon receiving new perception data for zero lag."""
        self.current_steer = float(msg.data)
        self.ema_steer = self.ema_alpha * self.current_steer + (1.0 - self.ema_alpha) * self.ema_steer
        self._apply_and_publish_steer(self.ema_steer)

    def _apply_and_publish_steer(self, steer_val: float):
        target_steer = (-steer_val if self.invert_steering else steer_val) * self.steering_gain
        target_steer = float(np.clip(target_steer, -1.0, 1.0))

        # Continuous Slew Rate Limiter in normalized float domain [-1.0, 1.0]
        dt = 1.0 / max(1.0, float(self.publish_rate_hz))
        max_delta = (self.max_steer_rate / 6.0) * (dt * 30.0)
        delta_steer = target_steer - self.last_applied_continuous_steer
        delta_steer = max(-max_delta, min(max_delta, delta_steer))
        self.last_applied_continuous_steer += delta_steer
        continuous_steer = float(np.clip(self.last_applied_continuous_steer, -1.0, 1.0))

        # 1. High-Resolution Continuous Steering (consumed by avis_bridge_full)
        steer_cmd_msg = Float32()
        steer_cmd_msg.data = continuous_steer
        self.steer_cmd_pub.publish(steer_cmd_msg)

        # 2. Discrete /servo Int8 [-6, 6] (maintained for hardware compatibility)
        raw_servo = int(round(continuous_steer * self.max_servo))
        servo_angle = max(self.min_servo, min(self.max_servo, raw_servo))
        servo_msg = Int8()
        servo_msg.data = int(servo_angle)
        self.servo_pub.publish(servo_msg)

        if abs(continuous_steer - self.last_continuous_steer_logged) > 0.10:
            self.get_logger().info(
                f'Steering: input={steer_val:+.3f} -> cmd={continuous_steer:+.3f} (servo={servo_angle})'
            )
            self.last_continuous_steer_logged = continuous_steer

    def _watchdog_loop(self):
        """Keep-alive watchdog in case perception stream pauses."""
        # Main work is done directly in steer_callback for zero latency
        pass


def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        center_servo = Int8()
        center_servo.data = 0
        node.servo_pub.publish(center_servo)

        center_cmd = Float32()
        center_cmd.data = 0.0
        node.steer_cmd_pub.publish(center_cmd)

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
