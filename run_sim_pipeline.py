#!/usr/bin/env python3
import sys
import os
import rclpy
from rclpy.executors import MultiThreadedExecutor

# مسیر سورس‌های جتسون
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "SelfDriving-jetson Source"))
from image_filter_node import ImageFilterNode
from lane_detection_node import LaneDetectionNode
from controller_node import ControllerNode

from avis_bridge_full import AvisFullBridgeNode


def main():
    print("=" * 60)
    print("  Starting Full Autonomous Pipeline for AVIS Engine  ")
    print("=" * 60)

    rclpy.init(args=sys.argv)

    bridge_node = AvisFullBridgeNode()
    filter_node = ImageFilterNode()
    lane_node = LaneDetectionNode()
    ctrl_node = ControllerNode()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(bridge_node)
    executor.add_node(filter_node)
    executor.add_node(lane_node)
    executor.add_node(ctrl_node)

    try:
        print("[INFO] All nodes active. Spinning executor...")
        executor.spin()
    except KeyboardInterrupt:
        print("\n[INFO] Stopping vehicle and shutting down...")
    finally:
        bridge_node.send_command(speed=0, steering=0)
        bridge_node.destroy_node()
        filter_node.destroy_node()
        lane_node.destroy_node()
        ctrl_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()