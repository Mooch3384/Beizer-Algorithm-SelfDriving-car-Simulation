#!/usr/bin/env python3
"""
Standalone Python Launcher for ROS 2 AvisEngine Lane Tracker Pipeline.
Executes AvisSimBridgeNode and LaneTrackerNode concurrently using rclpy.
"""

import sys
import os
import rclpy
from rclpy.executors import MultiThreadedExecutor

# Add workspace install path to sys.path
workspace_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(workspace_dir, "install", "avis_lane_tracker", "lib", "python3.12", "site-packages"))
sys.path.insert(0, "/opt/ros/jazzy/lib/python3.12/site-packages")

from avis_lane_tracker.avis_sim_bridge import AvisSimBridgeNode
from avis_lane_tracker.lane_tracker_node import LaneTrackerNode


def main():
    print("=" * 60)
    print("  Starting ROS 2 AvisEngine Online Lane Tracker & Bridge  ")
    print("=" * 60)

    rclpy.init(args=sys.argv)

    bridge_node = AvisSimBridgeNode()
    tracker_node = LaneTrackerNode()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(bridge_node)
    executor.add_node(tracker_node)

    try:
        print("[INFO] Multi-node executor spinning. Press Ctrl+C to stop.")
        executor.spin()
    except KeyboardInterrupt:
        print("\n[INFO] Keyboard Interrupt received. Shutting down ROS 2 nodes...")
    finally:
        executor.remove_node(bridge_node)
        executor.remove_node(tracker_node)
        bridge_node.destroy_node()
        tracker_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("[INFO] Clean shutdown complete.")


if __name__ == '__main__':
    main()
