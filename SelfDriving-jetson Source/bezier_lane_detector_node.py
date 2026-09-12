#!/usr/bin/env python3
"""
bezier_lane_detector_node.py
ROS 2 Inference & Lateral Control Node using Parametric Bézier Curves.

Replaces legacy image_filter_node and sliding-windows lane_detection_node:
1. Subscribes to /camera/image_raw (sensor_msgs/msg/Image, 512x512).
2. Deep Learning Path: Executes BézierLaneNet cubic curve inference when valid checkpoint (.pth) is provided.
3. Automatic CV Fallback: When no checkpoint is loaded or 0 lanes are detected, executes
   real-time HSV color segmentation (yellow/white lane thresholds) and multi-horizon centroid/moments
   extraction on the lower half of the frame.
4. Generates parametric cubic Bézier curves (Left, Right, Center) and computes analytical lateral deviation,
   heading error, and road curvature.
5. Applies calibrated lateral PID + curvature feedforward control (Kp=0.144, Ki=0.0, Kd=3.0, Kc=0.8)
   to actively steer through road curves.
6. Publishes /steering_value, /lane_status, /bezier_lane/path, and /debug_image.
"""

import sys
import os
import math
import time
import json
from collections import deque
from typing import Optional, Tuple, List

import cv2
import numpy as np

# ROS 2 Imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String, Int8
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped

# Ensure source and model paths are accessible
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "BezierLaneNet"))

from bezier_math import CubicBezier, compute_center_bezier, extract_control_metrics, get_perspective_lane_width

# Optional PyTorch import
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class BezierLaneDetectorNode(Node):
    def __init__(self):
        super().__init__('bezier_lane_detector_node')

        # Real-time QoS profiles
        # sensor_qos: RELIABLE publisher is compatible with both BEST_EFFORT and RELIABLE subscribers
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        rt_sub_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        # odom_qos: matching avis_bridge_full and car_point_visualizer (BEST_EFFORT, depth 10)
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ── Parameter Declarations ───────────────────────────────────────────
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('ckpt_path', '')
        self.declare_parameter('use_cuda', True)
        self.declare_parameter('model_input_width', 820)
        self.declare_parameter('model_input_height', 295)
        self.declare_parameter('max_lane', 4)
        self.declare_parameter('degree', 3)  # Cubic Bézier: 4 control points

        # Calibrated PID & Control Gains
        self.declare_parameter('kp', 0.70)
        self.declare_parameter('ki', 0.00)
        self.declare_parameter('kd', 1.00)
        self.declare_parameter('kc', 0.15)          # Curvature feedforward
        self.declare_parameter('k_heading', 0.30)   # Heading error damping
        self.declare_parameter('d_filter_alpha', 0.20)

        # Lookahead horizons (relative to image height)
        self.declare_parameter('lookahead_near_ratio', 0.85)
        self.declare_parameter('lookahead_mid_ratio', 0.65)
        self.declare_parameter('lookahead_far_ratio', 0.45)
        self.declare_parameter('expected_lane_width', 230.0)

        # Dead-reckoning & visualizer settings
        self.declare_parameter('dead_reckon_max_frames', 35)
        self.declare_parameter('show_display', False)
        self.declare_parameter('confidence_threshold', 0.30)

        self._load_parameters()
        self.add_on_set_parameters_callback(self._param_callback)

        # ── Subscribers & Publishers ─────────────────────────────────────────
        self.image_sub = self.create_subscription(
            Image, self.camera_topic, self.image_callback, rt_sub_qos)

        self.steer_pub = self.create_publisher(Float32, '/steering_value', sensor_qos)
        self.status_pub = self.create_publisher(String, '/lane_status', sensor_qos)
        self.debug_pub = self.create_publisher(Image, '/debug_image', sensor_qos)
        self.bezier_debug_pub = self.create_publisher(Image, '/bezier_lane/debug_image', sensor_qos)
        self.path_pub = self.create_publisher(Path, '/bezier_lane/path', sensor_qos)

        # ── Autonomy State Variables ─────────────────────────────────────────
        self.last_steer_time = None
        self.prev_error = None
        self.filtered_derivative = 0.0
        self.integral_error = 0.0
        self.last_good_steer = 0.0

        self.dr_mode = False
        self.dr_counter = 0
        self.mode = "CV_FALLBACK_TRACKING"

        # Frame timing
        self.fps_deque = deque(maxlen=20)
        self.inference_latency_ms = 0.0
        self.frame_count = 0

        # Cached previous good curves for smoothing
        self.prev_left_curve = None
        self.prev_right_curve = None

        # ── Initialize Deep Learning Model ───────────────────────────────────
        self.device = None
        self.model = None
        self.has_valid_weights = False
        self._init_model()

        self.get_logger().info(
            f"BezierLaneDetectorNode initialized (PyTorch: {TORCH_AVAILABLE}, Device: {self.device}, Valid Weights: {self.has_valid_weights})"
        )

    def _load_parameters(self):
        p = self.get_parameter
        self.camera_topic = p('camera_topic').value
        self.ckpt_path = p('ckpt_path').value
        self.use_cuda = p('use_cuda').value
        self.model_input_width = p('model_input_width').value
        self.model_input_height = p('model_input_height').value
        self.max_lane = p('max_lane').value
        self.degree = p('degree').value

        self.kp = p('kp').value
        self.ki = p('ki').value
        self.kd = p('kd').value
        self.kc = p('kc').value
        self.k_heading = p('k_heading').value
        self.d_filter_alpha = p('d_filter_alpha').value

        self.lookahead_near_ratio = p('lookahead_near_ratio').value
        self.lookahead_mid_ratio = p('lookahead_mid_ratio').value
        self.lookahead_far_ratio = p('lookahead_far_ratio').value
        self.expected_lane_width = p('expected_lane_width').value

        self.dead_reckon_max_frames = p('dead_reckon_max_frames').value
        self.show_display = p('show_display').value
        self.confidence_threshold = p('confidence_threshold').value

    def _param_callback(self, params):
        old_ckpt = self.ckpt_path
        for param in params:
            self.get_logger().info(f"Param updated: {param.name} = {param.value}")
        self._load_parameters()
        if self.ckpt_path != old_ckpt:
            self._init_model()
        return SetParametersResult(successful=True)

    def _init_model(self):
        """Initializes BézierLaneNet model if checkpoint weights are present, or sets up CV fallback."""
        if not TORCH_AVAILABLE:
            self.get_logger().warn("PyTorch is not available. Running Computer Vision Centroid Fallback.")
            self.has_valid_weights = False
            return

        if self.use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        # Check if a valid checkpoint is supplied
        if self.ckpt_path and os.path.isfile(self.ckpt_path):
            try:
                from models.custom_resnet import CustomResnet
                num_fc_nodes = (self.degree + 1) * 2 * self.max_lane
                feat_dim = 384

                self.model = CustomResnet(
                    feat_dim=feat_dim,
                    ckpt='',
                    max_lane=self.max_lane,
                    num_fc_nodes=num_fc_nodes
                )

                self.get_logger().info(f"Loading checkpoint weights from: {self.ckpt_path}")
                checkpoint = torch.load(self.ckpt_path, map_location=self.device)
                state_dict = checkpoint.get('state_dict', checkpoint)
                self.model.load_state_dict(state_dict, strict=False)
                self.model.to(self.device)
                self.model.eval()

                # Warm up
                dummy_input = torch.zeros((1, 3, self.model_input_height, self.model_input_width), device=self.device)
                with torch.no_grad():
                    _ = self.model(dummy_input)
                self.has_valid_weights = True
                self.get_logger().info("BézierLaneNet model loaded and warmed up successfully.")
            except Exception as e:
                self.get_logger().error(f"Failed to load checkpoint: {e}. Falling back to Computer Vision Centroid Tracking.")
                self.model = None
                self.has_valid_weights = False
        else:
            self.get_logger().info(
                "No valid checkpoint specified. Active perception mode: Automatic Computer Vision Centroid Fallback."
            )
            self.model = None
            self.has_valid_weights = False



    # ── Image Decoding & Encoding (Zero cv_bridge Dependency) ────────────────
    def _ros_image_to_cv2(self, msg: Image) -> Optional[np.ndarray]:
        """Converts sensor_msgs/Image to OpenCV BGR numpy array without depending on cv_bridge."""
        try:
            if msg.encoding in ('bgr8', 'rgb8'):
                im = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
                if msg.encoding == 'rgb8':
                    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                return im.copy()
            elif msg.encoding == 'mono8':
                im = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
                return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            else:
                im = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1))
                return im[:, :, :3].copy()
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return None

    def _cv2_to_ros_image(self, cv_img: np.ndarray, header) -> Image:
        """Converts OpenCV BGR image to sensor_msgs/Image without depending on cv_bridge."""
        msg = Image()
        msg.header = header
        msg.height = cv_img.shape[0]
        msg.width = cv_img.shape[1]
        msg.encoding = 'bgr8'
        msg.is_bigendian = 0
        msg.step = cv_img.shape[1] * 3
        msg.data = cv_img.tobytes()
        return msg

    # ── Main Perception & Control Pipeline ───────────────────────────────────
    def image_callback(self, msg: Image):
        t0 = time.time()
        frame = self._ros_image_to_cv2(msg)
        if frame is None:
            return

        orig_h, orig_w = frame.shape[:2]

        # 1. Detect Bézier Lanes (Deep Learning when weights present; CV Fallback otherwise)
        left_curve, right_curve, center_curve = self._detect_bezier_lanes(frame)

        # 2. Calculate Vehicle Control Metrics & Steering
        t_control_start = time.time()
        steer_cmd, metrics = self._calculate_steering(center_curve, orig_w, orig_h, t_control_start)

        # 3. Publish Normalized Steering & Lane Status
        steer_msg = Float32()
        steer_msg.data = float(steer_cmd)
        self.steer_pub.publish(steer_msg)

        status_dict = {
            "mode": self.mode,
            "detected": center_curve is not None,
            "recovering": self.dr_mode,
            "curvature": metrics.get("radius", -1.0) if metrics else -1.0,
            "signed_curvature": metrics.get("curvature", 0.0) if metrics else 0.0,
            "heading_err_deg": metrics.get("heading_err_deg", 0.0) if metrics else 0.0,
            "blended_err": metrics.get("blended_err", 0.0) if metrics else 0.0,
        }
        status_msg = String()
        status_msg.data = json.dumps(status_dict)
        self.status_pub.publish(status_msg)

        # 4. Publish RViz2 Path Trajectory
        if center_curve is not None:
            self._publish_path(center_curve, msg.header)

        # 5. Render Visual Debugger HUD & Publish (efficiently gated by active subscribers)
        self.frame_count += 1
        has_subscribers = (self.debug_pub.get_subscription_count() > 0 or
                           self.bezier_debug_pub.get_subscription_count() > 0 or
                           self.show_display)
        should_render_debug = has_subscribers and ((self.frame_count == 1) or (self.frame_count % 5 == 0))
        if should_render_debug:
            t_total = time.time() - t0
            fps = 1.0 / max(1e-4, t_total)
            self.fps_deque.append(fps)
            avg_fps = sum(self.fps_deque) / len(self.fps_deque)

            debug_frame = self._render_debugger(
                frame, left_curve, right_curve, center_curve, metrics, steer_cmd, avg_fps
            )
            debug_msg = self._cv2_to_ros_image(debug_frame, msg.header)
            self.debug_pub.publish(debug_msg)
            self.bezier_debug_pub.publish(debug_msg)

            if self.show_display:
                cv2.imshow("BézierLaneNet Real-Time Debugger", debug_frame)
                cv2.waitKey(1)

    def _detect_bezier_lanes(self, frame: np.ndarray) -> Tuple[Optional[CubicBezier], Optional[CubicBezier], Optional[CubicBezier]]:
        """
        Detects left, right, and center cubic Bézier curves.
        Uses BézierLaneNet when valid weights exist; automatically runs CV centroid fallback otherwise.
        """
        orig_h, orig_w = frame.shape[:2]

        # ── Branch A: Deep Learning Inference (When Checkpoint Available) ───
        if self.has_valid_weights and self.model is not None:
            try:
                t_infer_start = time.time()
                resized = cv2.resize(frame, (self.model_input_width, self.model_input_height))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                norm_img = (rgb - mean) / std

                tensor = torch.from_numpy(norm_img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    out_cls, out_ctrl = self.model(tensor)

                self.inference_latency_ms = (time.time() - t_infer_start) * 1000.0

                pred_lanes_count = int(torch.argmax(out_cls, dim=1).item())

                if pred_lanes_count > 0:
                    ctrl_data = out_ctrl.squeeze(0).cpu().numpy()
                    ctrl_points_all = ctrl_data.reshape((self.max_lane, self.degree + 1, 2))

                    scale_x = orig_w / float(self.model_input_width)
                    scale_y = orig_h / float(self.model_input_height)

                    detected_curves = []
                    for idx in range(min(pred_lanes_count, self.max_lane)):
                        pts = ctrl_points_all[idx].copy()
                        pts[:, 0] *= scale_x
                        pts[:, 1] *= scale_y
                        if np.all(np.isfinite(pts)):
                            detected_curves.append(CubicBezier(pts))

                    car_center_x = orig_w * 0.5
                    left_c, right_c = self._partition_lanes(detected_curves, car_center_x, orig_h)
                    if left_c is not None or right_c is not None:
                        center_c = compute_center_bezier(left_c, right_c, self.expected_lane_width)
                        self.mode = "VISION_TRACKING"
                        return left_c, right_c, center_c
            except Exception as e:
                self.get_logger().error(f"DL inference exception: {e}. Switching to CV Centroid Fallback.")

        # ── Branch B: Automatic Computer Vision / Thresholding Centroid Fallback ───
        left_c, right_c, center_c = self._cv_centroid_fallback(frame)
        if center_c is not None:
            self.mode = "CV_FALLBACK_TRACKING"
            return left_c, right_c, center_c

        return None, None, None

    def _partition_lanes(self, curves: List[CubicBezier], car_center_x: float, orig_h: float):
        """Partitions detected curves into left and right lanes closest to vehicle centerline."""
        left_candidates = []
        right_candidates = []

        for c in curves:
            t_bot = c.find_t_for_y(orig_h * 0.95)
            x_bot = c.eval(t_bot)[0]
            dist = x_bot - car_center_x
            if dist < 0:
                left_candidates.append((abs(dist), c))
            else:
                right_candidates.append((dist, c))

        left_curve = min(left_candidates, key=lambda item: item[0])[1] if left_candidates else None
        right_curve = min(right_candidates, key=lambda item: item[0])[1] if right_candidates else None
        return left_curve, right_curve

    def _cv_centroid_fallback(self, frame: np.ndarray) -> Tuple[Optional[CubicBezier], Optional[CubicBezier], Optional[CubicBezier]]:
        """
        Robust Computer Vision Centroid & Moments Fallback.
        Performs HSV color segmentation (yellow left lane, white right lane),
        masks out vehicle hood, extracts lane centroids across 4 vertical lookahead bands,
        and constructs parametric cubic Bézier curves with perspective continuity.
        """
        t_start = time.time()
        h, w = frame.shape[:2]
        car_x = w * 0.5

        # Process ROI (road is in lower portion y in [220, 512]) for fast execution
        roi_y_min = 220
        roi = frame[roi_y_min:, :]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 1. Color Segmentation
        # Yellow mask (left lane)
        yellow_mask_roi = cv2.inRange(hsv_roi, np.array([15, 55, 75], dtype=np.uint8), np.array([38, 255, 255], dtype=np.uint8))
        # White mask (right lane)
        white_mask_roi = cv2.inRange(hsv_roi, np.array([0, 0, 170], dtype=np.uint8), np.array([180, 55, 255], dtype=np.uint8))

        # Combine yellow and white lane markings into single robust road markings mask
        lane_mask_roi = cv2.bitwise_or(yellow_mask_roi, white_mask_roi)
        lane_mask = np.zeros((h, w), dtype=np.uint8)
        lane_mask[roi_y_min:, :] = lane_mask_roi

        # 2. Vehicle Hood Masking (mask out bottom center where hood is visible)
        lane_mask[400:512, 130:382] = 0

        # 3. Multi-Slice Lookahead Bands (Near to Far)
        bands = [
            (410, 460, 435.0),
            (350, 400, 375.0),
            (290, 340, 315.0),
            (230, 280, 255.0),
        ]

        left_pts = []
        right_pts = []
        center_pts = []

        prev_cx_l = None
        prev_cx_r = None

        for y_min, y_max, y_center in bands:
            slice_mask = lane_mask[y_min:y_max, :]
            lane_w = get_perspective_lane_width(y_center)

            # Left lane search window
            if prev_cx_l is None:
                search_l_min = 0
                search_l_max = int(car_x + 50)
            else:
                search_l_min = max(0, int(prev_cx_l - 70))
                search_l_max = min(int(car_x + 60), int(prev_cx_l + 70))

            mask_l = np.zeros_like(slice_mask)
            mask_l[:, search_l_min:search_l_max] = slice_mask[:, search_l_min:search_l_max]
            M_l = cv2.moments(mask_l)
            cx_l = float(M_l['m10'] / M_l['m00']) if M_l['m00'] > 20 else None

            # Right lane search window
            if prev_cx_r is not None:
                exp_r = prev_cx_r
            elif cx_l is not None:
                exp_r = cx_l + lane_w
            else:
                exp_r = car_x + 0.5 * lane_w

            search_r_min = max(int(car_x - 40), int(exp_r - 55))
            search_r_max = min(w, int(exp_r + 55))

            mask_r = np.zeros_like(slice_mask)
            mask_r[:, search_r_min:search_r_max] = slice_mask[:, search_r_min:search_r_max]
            M_r = cv2.moments(mask_r)
            cx_r = float(M_r['m10'] / M_r['m00']) if M_r['m00'] > 20 else None

            # Perspective width validation (reject far environmental white reflections)
            if cx_l is not None and cx_r is not None:
                if abs((cx_r - cx_l) - lane_w) > 0.35 * lane_w:
                    cx_c = cx_l + 0.5 * lane_w
                    cx_r = cx_l + lane_w
                else:
                    cx_c = 0.5 * (cx_l + cx_r)
            elif cx_l is not None:
                cx_c = cx_l + 0.5 * lane_w
            elif cx_r is not None:
                cx_c = cx_r - 0.5 * lane_w
            else:
                cx_c = None

            if cx_l is not None:
                left_pts.append([cx_l, y_center])
                prev_cx_l = cx_l
            if cx_r is not None:
                right_pts.append([cx_r, y_center])
                prev_cx_r = cx_r
            if cx_c is not None:
                center_pts.append([cx_c, y_center])

        self.inference_latency_ms = (time.time() - t_start) * 1000.0

        left_curve = None
        right_curve = None
        center_curve = None

        if len(left_pts) >= 3:
            indices = np.linspace(0, len(left_pts) - 1, 4)
            c_pts = np.array([left_pts[int(round(i))] for i in indices], dtype=np.float64)
            left_curve = CubicBezier(c_pts)

        if len(right_pts) >= 3:
            indices = np.linspace(0, len(right_pts) - 1, 4)
            c_pts = np.array([right_pts[int(round(i))] for i in indices], dtype=np.float64)
            right_curve = CubicBezier(c_pts)

        if len(center_pts) >= 3:
            indices = np.linspace(0, len(center_pts) - 1, 4)
            c_pts = np.array([center_pts[int(round(i))] for i in indices], dtype=np.float64)
            center_curve = CubicBezier(c_pts)
        else:
            center_curve = compute_center_bezier(left_curve, right_curve, self.expected_lane_width)

        return left_curve, right_curve, center_curve

    def _calculate_steering(self,
                            center_curve: Optional[CubicBezier],
                            image_width: int,
                            image_height: int,
                            current_time_sec: float) -> Tuple[float, dict]:
        """
        Calculates normalized steering command [-1.0, 1.0] using:
        1. Multi-lookahead lateral error (Near, Mid, Far).
        2. Analytical Bézier curvature feedforward.
        3. Heading angle error damping.
        4. Calibrated lateral PID controller.
        """
        if self.last_steer_time is None:
            dt = 1.0 / 30.0
        else:
            dt = current_time_sec - self.last_steer_time
            if dt <= 0.001 or dt > 0.5:
                dt = 1.0 / 30.0
        self.last_steer_time = current_time_sec

        # Handle Dead Reckoning if lane lost
        if center_curve is None:
            self.dr_counter += 1
            self.integral_error = 0.0
            if not self.dr_mode:
                self.dr_mode = True
                self.mode = "DEAD_RECKONING"

            decay = max(0.0, 1.0 - self.dr_counter / float(self.dead_reckon_max_frames))
            if self.dr_counter < self.dead_reckon_max_frames:
                steer = self.last_good_steer * decay
            else:
                steer = 0.0
                self.mode = "LANE_LOST"
            return float(np.clip(steer, -1.0, 1.0)), {}

        # Reset dead reckoning
        self.dr_mode = False
        self.dr_counter = 0

        # Fixed, robust lookahead horizons (Near, Mid, Far)
        near_ratio = self.lookahead_near_ratio   # default 0.85 (y = 435)
        mid_ratio = self.lookahead_mid_ratio     # default 0.65 (y = 332)
        far_ratio = self.lookahead_far_ratio     # default 0.45 (y = 230)

        # Balanced lookahead weights: near tracking + mid stability + far curve anticipation
        lookahead_weights = (0.40, 0.40, 0.20)

        metrics = extract_control_metrics(
            center_curve,
            image_width=image_width,
            image_height=image_height,
            lookahead_near_ratio=near_ratio,
            lookahead_mid_ratio=mid_ratio,
            lookahead_far_ratio=far_ratio,
            car_x_offset=image_width * 0.5,
            weights=lookahead_weights
        )

        err = metrics["blended_err"]
        radius = metrics["radius"]
        curvature = metrics["curvature"]
        heading_err = metrics["heading_err_rad"]

        # ── Lateral Control Formulation ──────────────────────────────────────
        # 1. Proportional term:
        # Scale: kp * (err / 100.0) -> with kp=0.70:
        #  10 px error -> 0.07 steer
        #  30 px error -> 0.21 steer
        #  50 px error -> 0.35 steer
        p_term = self.kp * (err / 100.0)

        # 2. Derivative term with low-pass filtering and first-frame protection:
        if self.prev_error is None:
            self.prev_error = err
            raw_derivative = 0.0
        else:
            raw_derivative = (err - self.prev_error) / dt

        self.filtered_derivative = (self.d_filter_alpha * raw_derivative +
                                    (1.0 - self.d_filter_alpha) * self.filtered_derivative)
        d_term = float(np.clip(self.kd * (self.filtered_derivative * 0.0004), -0.12, 0.12))

        # 3. Anti-windup Integral term:
        if abs(err) < 40.0:
            self.integral_error = max(-20.0, min(20.0, self.integral_error + err * dt))
        else:
            self.integral_error *= 0.5
        i_term = self.ki * self.integral_error

        # 4. Gentle Curvature Feedforward (does not saturate or fight tracking):
        cs = 0.0
        if radius < 900.0:
            direction = 1.0 if curvature > 0 else (-1.0 if curvature < 0 else 0.0)
            curv_factor = min(1.0, 350.0 / max(radius, 50.0))
            cs = direction * self.kc * curv_factor

        # 5. Heading Error Damping:
        heading_term = self.k_heading * heading_err * 0.30

        # Combined normalized steering [-1.0, 1.0]
        raw_steer = p_term + d_term + i_term + cs + heading_term
        steer = float(np.clip(raw_steer, -1.0, 1.0))

        self.prev_error = err
        self.last_good_steer = steer
        return steer, metrics

    def _publish_path(self, center_curve: CubicBezier, header):
        """Publishes Bézier target path as nav_msgs/msg/Path for RViz2."""
        path_msg = Path()
        path_msg.header = header
        path_msg.header.frame_id = 'base_link'

        ts = np.linspace(0.0, 1.0, 20)
        pts = center_curve.eval_multi(ts)

        for pt in pts:
            pose = PoseStamped()
            pose.header = header
            pose.header.frame_id = 'base_link'
            pose.pose.position.x = float((512.0 - pt[1]) * 0.01)
            pose.pose.position.y = float((256.0 - pt[0]) * 0.01)
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

    # ── Real-Time Visual Debugger Overlay ────────────────────────────────────
    def _render_debugger(self,
                         frame: np.ndarray,
                         left_curve: Optional[CubicBezier],
                         right_curve: Optional[CubicBezier],
                         center_curve: Optional[CubicBezier],
                         metrics: dict,
                         steer_cmd: float,
                         fps: float) -> np.ndarray:
        vis = frame.copy()
        h, w = vis.shape[:2]

        # 1. Draw Left Lane (Red #FF3333)
        if left_curve is not None:
            pts_l = left_curve.sample_points(64)
            cv2.polylines(vis, [pts_l], isClosed=False, color=(50, 50, 240), thickness=3, lineType=cv2.LINE_AA)
            for i, p in enumerate(left_curve.pts):
                cv2.circle(vis, (int(p[0]), int(p[1])), 5, (0, 0, 255), -1)
                cv2.putText(vis, f"L{i}", (int(p[0]) + 6, int(p[1]) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # 2. Draw Right Lane (Blue #3388FF)
        if right_curve is not None:
            pts_r = right_curve.sample_points(64)
            cv2.polylines(vis, [pts_r], isClosed=False, color=(240, 130, 50), thickness=3, lineType=cv2.LINE_AA)
            for i, p in enumerate(right_curve.pts):
                cv2.circle(vis, (int(p[0]), int(p[1])), 5, (255, 100, 0), -1)
                cv2.putText(vis, f"R{i}", (int(p[0]) + 6, int(p[1]) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 100, 0), 1)

        # 3. Draw Center Target Path (Bright Cyan #00FFFF)
        if center_curve is not None:
            pts_c = center_curve.sample_points(64)
            cv2.polylines(vis, [pts_c], isClosed=False, color=(255, 255, 0), thickness=3, lineType=cv2.LINE_AA)
            for i, p in enumerate(center_curve.pts):
                cv2.circle(vis, (int(p[0]), int(p[1])), 4, (0, 255, 255), -1)

            # Lookahead target points
            if "pt_near" in metrics:
                cv2.circle(vis, metrics["pt_near"], 7, (0, 255, 0), -1)
                cv2.putText(vis, "Near", (metrics["pt_near"][0] + 10, metrics["pt_near"][1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            if "pt_mid" in metrics:
                cv2.circle(vis, metrics["pt_mid"], 7, (0, 200, 255), -1)
                cv2.putText(vis, "Mid", (metrics["pt_mid"][0] + 10, metrics["pt_mid"][1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
            if "pt_far" in metrics:
                cv2.circle(vis, metrics["pt_far"], 7, (0, 100, 255), -1)
                cv2.putText(vis, "Far", (metrics["pt_far"][0] + 10, metrics["pt_far"][1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1)

        # 4. Vehicle Center Guide Line
        car_x = int(w * 0.5)
        cv2.line(vis, (car_x, h), (car_x, int(h * 0.5)), (0, 255, 0), 1, cv2.LINE_AA)

        # 5. Diagnostic HUD Banner
        hud_bg = np.zeros((120, 320, 3), dtype=np.uint8)
        hud_bg[:] = (20, 20, 20)
        cv2.addWeighted(vis[10:130, 10:330], 0.3, hud_bg, 0.7, 0, vis[10:130, 10:330])

        e_near = metrics.get("e_near", 0.0)
        e_mid = metrics.get("e_mid", 0.0)
        psi_deg = metrics.get("heading_err_deg", 0.0)
        rad = metrics.get("radius", 9999.0)

        mode_color = (0, 255, 0) if self.mode == "VISION_TRACKING" else ((0, 220, 255) if self.mode == "CV_FALLBACK_TRACKING" else (0, 50, 255))

        cv2.putText(vis, f"BézierLaneNet Autonomy", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, f"FPS: {fps:.1f} | Latency: {self.inference_latency_ms:.1f}ms", (20, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, f"e_near: {e_near:+.1f}px | e_mid: {e_mid:+.1f}px", (20, 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 255, 200), 1, cv2.LINE_AA)
        cv2.putText(vis, f"Heading Err: {psi_deg:+.1f} deg", (20, 84),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, f"Radius: {rad:.0f}m | Steer: {steer_cmd:+.2f}", (20, 102),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, f"Mode: {self.mode}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, mode_color, 1, cv2.LINE_AA)

        return vis


def main(args=None):
    rclpy.init(args=args)
    node = BezierLaneDetectorNode()
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
