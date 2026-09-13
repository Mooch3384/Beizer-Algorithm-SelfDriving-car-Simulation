#!/usr/bin/env python3
"""
run_sim_pipeline_bezier.py
Unified Autonomous Simulator Pipeline for AVIS Engine using BézierLaneNet.

Features:
- Automated simulator process lifecycle management (starts cleanly, terminates completely on Ctrl+C)
- Seamless BézierLaneNet neural inference (with auto-detection of trained weights)
- High-speed fallback to analytical CV Bézier curve fitting
- Calibrated lateral PID + curvature feedforward control
- Clear operator guidance for AvisEngine track selection and autonomous mode
"""

import sys
import os
import time
import socket
import signal
import argparse
import subprocess
import rclpy
from rclpy.executors import MultiThreadedExecutor

workspace_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.join(workspace_dir, "SelfDriving-jetson Source")
virtual_modelling_dir = os.path.join(workspace_dir, "Virtual-Modelling")
sys.path.insert(0, source_dir)
sys.path.insert(0, virtual_modelling_dir)
sys.path.insert(0, workspace_dir)

from avis_bridge_full import AvisFullBridgeNode
from bezier_lane_detector_node import BezierLaneDetectorNode
from controller_node import ControllerNode
from car_point_visualizer import CarPointVisualizer


def is_port_listening(port=25001):
    """
    Check if port is listening by reading Linux kernel TCP tables (/proc/net/tcp).
    Crucial: Does NOT initiate a TCP connection, preventing consumption of
    AvisEngine's single-call AcceptTcpClient() before the real bridge connects.
    """
    hex_port = f":{port:04X}"
    for path in ('/proc/net/tcp', '/proc/net/tcp6'):
        try:
            with open(path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 3 and parts[1].endswith(hex_port) and parts[3] == '0A':
                        return True
        except Exception:
            pass
    return False


def ensure_simulator_running(sim_executable: str):
    if is_port_listening(25001):
        print("[INFO] AVIS Engine is already running and listening on port 25001.")
        return None

    if not os.path.isfile(sim_executable):
        print(f"[WARN] Simulator binary not found at {sim_executable}. Waiting for manual launch...")
        return None

    print(f"[INFO] Launching AVIS Engine simulator: {sim_executable} ...")
    sim_dir = os.path.dirname(sim_executable)
    proc = subprocess.Popen(
        [sim_executable, "-logfile", "sim_runtime.log"],
        cwd=sim_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid
    )

    print("\n" + "=" * 68)
    print("  [راهنمای شبیه‌ساز AvisEngine]")
    print("  ۱. در پنجره شبیه‌ساز، پیست مورد نظر (مثلاً Urban Track 1) را انتخاب کنید.")
    print("  ۲. سرعت دلخواه خودرو را از طریق اسلایدر Top Speed در پنل شبیه‌ساز تنظیم کنید.")
    print("  ۳. دکمه 'Start Server' را بزنید تا ارتباط فعال شود.")
    print("  (کنترل خودران هوشمند خطوط را تشخیص داده و خودرو را در مرکز مسیر هدایت می‌کند)")
    print("=" * 68 + "\n")

    print("[INFO] Waiting for simulator TCP server on port 25001 (click 'Start Server' in sim)...")
    start_t = time.time()
    while time.time() - start_t < 60.0:
        if is_port_listening(25001):
            print("[INFO] Simulator TCP server detected and ready (port 25001 listening)!\n")
            return proc
        time.sleep(0.5)

    print("[WARN] Port 25001 wait timeout. Proceeding with pipeline...")
    return proc


def kill_all_simulators(proc=None):
    if proc is not None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    try:
        subprocess.run(["killall", "-9", "AVISEngine.x86_64"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="BézierLaneNet Autonomous Pipeline for AVIS Engine")
    parser.add_argument("--ckpt", type=str, default="", help="Path to BézierLaneNet model checkpoint (.pth)")
    parser.add_argument("--no_sim_launch", action="store_true", help="Do not auto-launch simulator")
    parser.add_argument("--rviz", action="store_true", help="Automatically launch RViz2 with preconfigured 3D visualization")
    args, unknown = parser.parse_known_args()

    print("=" * 68)
    print("  Starting BézierLaneNet Autonomous Pipeline for AVIS Engine Sim  ")
    print("=" * 68)

    # Check for default trained checkpoint if none specified
    ckpt_path = args.ckpt
    default_ckpt = os.path.join(workspace_dir, "weights", "bezier_avis_best.pth")
    if not ckpt_path and os.path.isfile(default_ckpt):
        ckpt_path = default_ckpt
        print(f"[INFO] Auto-detected trained checkpoint: {ckpt_path}")

    # Ensure simulator is up
    sim_proc = None
    if not args.no_sim_launch:
        sim_path = os.path.join(workspace_dir, "Linux", "AVISEngine.x86_64")
        sim_proc = ensure_simulator_running(sim_path)

    # Optional RViz2 launch
    rviz_proc = None
    if args.rviz:
        rviz_config_path = os.path.join(workspace_dir, "config", "car_simulation.rviz")
        if os.path.isfile(rviz_config_path):
            print(f"[INFO] Launching RViz2 with configuration: {rviz_config_path} ...")
            rviz_proc = subprocess.Popen(
                ["rviz2", "-d", rviz_config_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            print("[INFO] Launching default RViz2 ...")
            rviz_proc = subprocess.Popen(["rviz2"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    rclpy.init(args=sys.argv)

    # Configure parameters
    bridge_node = AvisFullBridgeNode()

    # Pass checkpoint parameter to detector if available
    detector_node = BezierLaneDetectorNode()
    if ckpt_path:
        detector_node.set_parameters([
            rclpy.parameter.Parameter('ckpt_path', rclpy.Parameter.Type.STRING, ckpt_path)
        ])

    controller_node = ControllerNode()
    visualizer_node = CarPointVisualizer()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(bridge_node)
    executor.add_node(detector_node)
    executor.add_node(controller_node)
    executor.add_node(visualizer_node)

    try:
        print("[INFO] All 4 nodes initialized under MultiThreadedExecutor:")
        print("       1. AvisFullBridgeNode (TCP 127.0.0.1:25001 <-> ROS 2, High Throughput Bridge, TF map->odom->base_link)")
        print(f"       2. BezierLaneDetectorNode (Checkpoint: '{ckpt_path}' or CV Fallback, Calibrated Tracking)")
        print("       3. ControllerNode (Continuous Actuation, Low-Latency Lateral Control)")
        print("       4. CarPointVisualizer (3D Point-Robot Model, 50m Planned Trajectory, RViz2 Visualizer)")
        print("[INFO] Spinning pipeline executor. Press Ctrl+C to stop.\n")
        executor.spin()
    except KeyboardInterrupt:
        print("\n[INFO] Keyboard Interrupt received. Stopping vehicle and shutting down...")
    finally:
        try:
            bridge_node.send_command(speed=0, steering=0)
            time.sleep(0.1)
        except Exception:
            pass

        bridge_node.destroy_node()
        detector_node.destroy_node()
        controller_node.destroy_node()
        visualizer_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

        if rviz_proc is not None:
            try:
                rviz_proc.terminate()
            except Exception:
                pass

        # Cleanly kill simulator
        print("[INFO] Closing AVIS Engine simulator process...")
        kill_all_simulators(sim_proc)

        print("[INFO] Clean shutdown complete.")


if __name__ == '__main__':
    main()
