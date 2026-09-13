#!/usr/bin/env python3
"""
test_node_integration.py
Tests BezierLaneDetectorNode instantiation, synthetic frame processing,
and topic publishing without requiring the live simulator.
"""

import sys
import os
import time
import json
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String
from nav_msgs.msg import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "SelfDriving-jetson Source"))
from bezier_lane_detector_node import BezierLaneDetectorNode


class VerificationSubscriber(Node):
    def __init__(self):
        super().__init__('verification_subscriber')
        self.received_steering = None
        self.received_status = None
        self.received_debug_img = None
        self.received_path = None
        self.received_planned_path = None

        self.create_subscription(Float32, '/steering_value', self.steer_cb, 10)
        self.create_subscription(String, '/lane_status', self.status_cb, 10)
        self.create_subscription(Image, '/debug_image', self.debug_cb, 10)
        self.create_subscription(Path, '/bezier_lane/path', self.path_cb, 10)
        self.create_subscription(Path, '/planned_trajectory', self.planned_path_cb, 10)

    def steer_cb(self, msg: Float32):
        self.received_steering = msg.data

    def status_cb(self, msg: String):
        self.received_status = json.loads(msg.data)

    def debug_cb(self, msg: Image):
        self.received_debug_img = (msg.width, msg.height)

    def path_cb(self, msg: Path):
        self.received_path = len(msg.poses)

    def planned_path_cb(self, msg: Path):
        self.received_planned_path = msg.poses


def run_test():
    print("Testing BezierLaneDetectorNode integration...")
    rclpy.init()

    detector = BezierLaneDetectorNode()
    verifier = VerificationSubscriber()

    # Create a synthetic 512x512 road frame with curved lane lines
    test_img = np.zeros((512, 512, 3), dtype=np.uint8)
    test_img[:] = (60, 60, 60)  # Asphalt grey

    # Draw left lane (white) and right lane (yellow)
    cv2.line(test_img, (140, 512), (200, 260), (255, 255, 255), 8)
    cv2.line(test_img, (370, 512), (310, 260), (0, 220, 220), 8)

    # Convert to ROS Image message
    img_msg = Image()
    img_msg.header.stamp = detector.get_clock().now().to_msg()
    img_msg.header.frame_id = 'camera_link'
    img_msg.height = 512
    img_msg.width = 512
    img_msg.encoding = 'bgr8'
    img_msg.step = 512 * 3
    img_msg.data = test_img.tobytes()

    # Direct callback execution to test processing pipeline
    detector.image_callback(img_msg)

    # Spin briefly to allow verifier subscriber to process published messages
    for _ in range(10):
        rclpy.spin_once(detector, timeout_sec=0.05)
        rclpy.spin_once(verifier, timeout_sec=0.05)

    print(f"Steering received: {verifier.received_steering}")
    print(f"Status received: {verifier.received_status}")
    print(f"Debug image received: {verifier.received_debug_img}")
    print(f"Legacy path poses count: {verifier.received_path}")
    print(f"Metric planned trajectory poses count: {len(verifier.received_planned_path) if verifier.received_planned_path else None}")

    assert verifier.received_steering is not None, "Steering message not received!"
    assert -1.0 <= verifier.received_steering <= 1.0, f"Steering {verifier.received_steering} out of range [-1, 1]!"
    assert verifier.received_status is not None, "Status message not received!"
    assert "target_metric_x" in verifier.received_status, "target_metric_x missing from status!"
    assert verifier.received_debug_img == (512, 512), f"Debug image resolution mismatch: {verifier.received_debug_img}"
    assert verifier.received_path is not None and verifier.received_path > 0, "Bézier Path not received!"
    assert verifier.received_planned_path is not None and len(verifier.received_planned_path) > 0, "Planned trajectory not received!"
    
    # Verify metric coordinates in base_link (X positive forward)
    first_pt = verifier.received_planned_path[0].pose.position
    last_pt = verifier.received_planned_path[-1].pose.position
    print(f"Planned trajectory starts at X={first_pt.x:.2f}m, Y={first_pt.y:.2f}m")
    print(f"Planned trajectory extends to X={last_pt.x:.2f}m, Y={last_pt.y:.2f}m")
    assert first_pt.x >= 0.0 and last_pt.x > first_pt.x, "Planned trajectory should extend forward in +X!"
    assert last_pt.x >= 12.0, f"Planned trajectory should extend forward, got {last_pt.x}m"

    detector.destroy_node()
    verifier.destroy_node()
    rclpy.shutdown()
    print("\n✓ BezierLaneDetectorNode ROS 2 integration verification PASSED!")


if __name__ == '__main__':
    run_test()
