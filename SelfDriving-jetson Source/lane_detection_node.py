#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String, Int8
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import numpy as np
from collections import deque
import json
import math
import time


class LaneTracker:
    PREVIEW_NEAR_RATIO = 0.85
    PREVIEW_FAR_RATIO = 0.65
    DEAD_RECKON_MAX_FRAMES = 40

    def __init__(self, image_width, image_height):
        self.image_width = image_width
        self.image_height = image_height
        self.left_fit = None
        self.right_fit = None
        self.left_fit_history = deque(maxlen=5)
        self.right_fit_history = deque(maxlen=5)
        self.lane_width = int(image_width * 0.75)
        self.detected = False
        self.left_detected = False
        self.right_detected = False
        self.prev_error = 0.0

        self.kp = 0.144
        self.kd = 3.0
        self.kc = 0.8

        self.roi_expand_factor = 1.0
        self.MAX_ROI_EXPAND = 1.5
        self.last_good_steer = 0.0
        self.blind_counter = 0
        self.last_good_left_fit = None
        self.last_good_right_fit = None
        self.confidence_left = 0.0
        self.confidence_right = 0.0
        self.curvature_radius = float('inf')
        self.curve_direction = 0
        self.low_confidence_counter = 0
        self.LOW_CONF_THRESHOLD = 2
        self.MIN_CONFIDENCE = 0.35
        self.prev_left_x_bottom = None
        self.prev_right_x_bottom = None
        self.MAX_JUMP_PIXELS = 80
        self.MIN_POINTS = 50
        self.force_sliding_window_counter = 0
        self.FORCE_SLIDING_WINDOW_INTERVAL = 10
        self.was_in_heavy_turn = False
        self.last_turn_direction = 0
        self.recovery_mode = False
        self.recovery_counter = 0
        self.MAX_RECOVERY_FRAMES = 30
        self.max_pixel_offset = 100

        self.nwindows = 6
        self.minpix = 15
        self.base_margin = 60
        self.max_empty = 4

        self.prior_margin = 50
        self.prior_min_points = 150

        self.use_prior = True
        self.use_sliding_window = False
        self.force_sw = False

        self.window_expand_per_empty = 15
        self.max_margin_base = 120

        self.max_curvature_a = 0.005

        self._sw_bootstrapped = False

        self.viz_histogram = None
        self.viz_method = "sliding_window"
        self.viz_left_windows = []
        self.viz_right_windows = []
        self.viz_left_pixels = (np.array([]), np.array([]))
        self.viz_right_pixels = (np.array([]), np.array([]))

        self.dr_mode = False
        self.dr_counter = 0
        self.dr_last_steer = 0.0
        self.dr_last_curvature = float('inf')
        self.dr_last_curve_dir = 0
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_theta = 0.0
        self.last_odom_x = None
        self.last_odom_y = None
        self.last_odom_theta = None
        self.mode = "VISION_TRACKING"

    def reset_tracker(self, reason=""):
        self.left_fit = None
        self.right_fit = None
        self.left_fit_history.clear()
        self.right_fit_history.clear()
        self.detected = False
        self.left_detected = False
        self.right_detected = False
        self.confidence_left = 0.0
        self.confidence_right = 0.0
        self.low_confidence_counter = 0
        self.prev_left_x_bottom = None
        self.prev_right_x_bottom = None
        self.recovery_mode = False
        self.roi_expand_factor = 1.0
        self._sw_bootstrapped = False
        self.viz_histogram = None
        self.viz_method = "sliding_window"
        self.viz_left_windows = []
        self.viz_right_windows = []
        self.viz_left_pixels = (np.array([]), np.array([]))
        self.viz_right_pixels = (np.array([]), np.array([]))

    def update_odom(self, x, y, theta):
        self.last_odom_x = x
        self.last_odom_y = y
        self.last_odom_theta = theta
        self.odom_x = x
        self.odom_y = y
        self.odom_theta = theta

    def detect_lane_jump(self, new_left_fit, new_right_fit):
        y_eval = self.image_height
        jump = False
        if new_left_fit is not None and self.prev_left_x_bottom is not None:
            if abs(np.polyval(new_left_fit, y_eval) - self.prev_left_x_bottom) > self.MAX_JUMP_PIXELS:
                jump = True
        if new_right_fit is not None and self.prev_right_x_bottom is not None:
            if abs(np.polyval(new_right_fit, y_eval) - self.prev_right_x_bottom) > self.MAX_JUMP_PIXELS:
                jump = True
        return jump

    def update_position_tracking(self):
        y_eval = self.image_height
        if self.left_fit is not None:
            self.prev_left_x_bottom = np.polyval(self.left_fit, y_eval)
        if self.right_fit is not None:
            self.prev_right_x_bottom = np.polyval(self.right_fit, y_eval)

    def find_lanes_sliding_window(self, binary_warped):
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 3:, :], axis=0)
        if histogram.max() > 0:
            histogram = cv2.GaussianBlur(histogram.astype(np.float32), (51, 1), 0).flatten()
        midpoint = int(histogram.shape[0] / 2)
        center_margin = int(self.image_width * 0.08)
        histogram[max(0, midpoint - center_margin):min(len(histogram), midpoint + center_margin)] = 0

        leftx_base = np.argmax(histogram[:midpoint]) if midpoint > 0 else 0
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint if midpoint < len(histogram) else self.image_width - 1

        nwindows = self.nwindows
        base_margin = self.base_margin
        margin = int(base_margin * self.roi_expand_factor)
        minpix = self.minpix
        window_height = int(binary_warped.shape[0] // nwindows)

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        left_x_positions = [leftx_current]
        left_y_positions = [binary_warped.shape[0]]
        right_x_positions = [rightx_current]
        right_y_positions = [binary_warped.shape[0]]

        left_empty_count = 0
        right_empty_count = 0
        MAX_EMPTY = self.max_empty

        self.viz_left_windows = []
        self.viz_right_windows = []
        self.viz_histogram = histogram

        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_y_center = (win_y_low + win_y_high) // 2

            max_m = int(self.max_margin_base * self.roi_expand_factor)
            lm = min(margin + left_empty_count * self.window_expand_per_empty, max_m)
            rm = min(margin + right_empty_count * self.window_expand_per_empty, max_m)

            xl_lo = max(0, leftx_current - lm)
            xl_hi = min(binary_warped.shape[1], leftx_current + lm)
            xr_lo = max(0, rightx_current - rm)
            xr_hi = min(binary_warped.shape[1], rightx_current + rm)

            self.viz_left_windows.append((xl_lo, xl_hi, win_y_low, win_y_high))
            self.viz_right_windows.append((xr_lo, xr_hi, win_y_low, win_y_high))

            if left_empty_count < MAX_EMPTY:
                gl = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                      (nonzerox >= xl_lo) & (nonzerox < xl_hi)).nonzero()[0]
            else:
                gl = np.array([], dtype=int)

            if right_empty_count < MAX_EMPTY:
                gr = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                      (nonzerox >= xr_lo) & (nonzerox < xr_hi)).nonzero()[0]
            else:
                gr = np.array([], dtype=int)

            left_lane_inds.append(gl)
            right_lane_inds.append(gr)

            if len(gl) > minpix:
                leftx_current = int(np.mean(nonzerox[gl]))
                left_x_positions.append(leftx_current)
                left_y_positions.append(win_y_center)
                left_empty_count = 0
            else:
                left_empty_count += 1
                if len(left_x_positions) >= 2:
                    dx = left_x_positions[-1] - left_x_positions[-2]
                    dy = left_y_positions[-1] - left_y_positions[-2]
                    if dy != 0:
                        leftx_current = max(0, min(binary_warped.shape[1] - 1,
                                                    leftx_current + int(dx / dy * window_height)))

            if len(gr) > minpix:
                rightx_current = int(np.mean(nonzerox[gr]))
                right_x_positions.append(rightx_current)
                right_y_positions.append(win_y_center)
                right_empty_count = 0
            else:
                right_empty_count += 1
                if len(right_x_positions) >= 2:
                    dx = right_x_positions[-1] - right_x_positions[-2]
                    dy = right_y_positions[-1] - right_y_positions[-2]
                    if dy != 0:
                        rightx_current = max(0, min(binary_warped.shape[1] - 1,
                                                     rightx_current + int(dx / dy * window_height)))

        left_lane_inds = np.concatenate(left_lane_inds).astype(np.intp) if left_lane_inds else np.array([], dtype=np.intp)
        right_lane_inds = np.concatenate(right_lane_inds).astype(np.intp) if right_lane_inds else np.array([], dtype=np.intp)

        lx = nonzerox[left_lane_inds] if len(left_lane_inds) > 0 else np.array([])
        ly = nonzeroy[left_lane_inds] if len(left_lane_inds) > 0 else np.array([])
        rx = nonzerox[right_lane_inds] if len(right_lane_inds) > 0 else np.array([])
        ry = nonzeroy[right_lane_inds] if len(right_lane_inds) > 0 else np.array([])

        self.viz_left_pixels = (lx, ly)
        self.viz_right_pixels = (rx, ry)
        return lx, ly, rx, ry

    def find_lanes_from_prior(self, binary_warped):
        margin = self.prior_margin
        nz = binary_warped.nonzero()
        nzy, nzx = np.array(nz[0]), np.array(nz[1])
        lx, ly, rx, ry = np.array([]), np.array([]), np.array([]), np.array([])

        if self.left_fit is not None:
            li = ((nzx > (self.left_fit[0] * nzy**2 + self.left_fit[1] * nzy + self.left_fit[2] - margin)) &
                  (nzx < (self.left_fit[0] * nzy**2 + self.left_fit[1] * nzy + self.left_fit[2] + margin)))
            lx, ly = nzx[li], nzy[li]

        if self.right_fit is not None:
            ri = ((nzx > (self.right_fit[0] * nzy**2 + self.right_fit[1] * nzy + self.right_fit[2] - margin)) &
                  (nzx < (self.right_fit[0] * nzy**2 + self.right_fit[1] * nzy + self.right_fit[2] + margin)))
            rx, ry = nzx[ri], nzy[ri]

        self.viz_left_windows = []
        self.viz_right_windows = []
        self.viz_histogram = None
        self.viz_left_pixels = (lx, ly)
        self.viz_right_pixels = (rx, ry)
        return lx, ly, rx, ry

    def fit_polynomial(self, x, y):
        if len(x) < self.MIN_POINTS:
            return None, 0.0
        try:
            w = y / self.image_height
            fit = np.polyfit(y, x, 2, w=w)
            res = np.mean(np.abs(x - np.polyval(fit, y)))
            return fit, max(0.0, 1.0 - res / 50.0)
        except Exception:
            return None, 0.0

    def smooth_fit(self, new_fit, history, confidence):
        if new_fit is None:
            return np.mean(history, axis=0) if len(history) > 0 else None
        if confidence > self.MIN_CONFIDENCE:
            history.append(new_fit)
        else:
            return np.mean(history, axis=0) if len(history) > 0 else new_fit
        if len(history) == 0:
            return new_fit
        w = np.linspace(0.5, 1.0, len(history))
        w /= w.sum()
        return np.average(history, axis=0, weights=w)

    def validate_lanes(self, lf, rf):
        if lf is None and rf is None:
            return False, False
        ye = self.image_height
        lv, rv = True, True

        if lf is not None:
            lb = np.polyval(lf, ye)
            lt = np.polyval(lf, 0)
            if lb < -120 or lb > self.image_width * 0.80:
                lv = False
            if abs(lf[0]) > 0.015:  # Allow sharper curves (was 0.005)
                lv = False
            if lt < lb - self.image_width * 0.6:
                lv = False

        if rf is not None:
            rb = np.polyval(rf, ye)
            rt = np.polyval(rf, 0)
            if rb > self.image_width + 120 or rb < self.image_width * 0.20:
                rv = False
            if abs(rf[0]) > 0.015:  # Allow sharper curves (was 0.005)
                rv = False
            if rt > rb + self.image_width * 0.6:
                rv = False

        if lf is not None and rf is not None and lv and rv:
            wb = np.polyval(rf, ye) - np.polyval(lf, ye)
            wt = np.polyval(rf, 0) - np.polyval(lf, 0)
            if wb < self.lane_width * 0.35 or wb > self.lane_width * 1.8:
                if self.confidence_left > self.confidence_right:
                    rv = False
                else:
                    lv = False
            if abs(wt - wb) > self.lane_width * 0.8:
                if self.confidence_left > self.confidence_right:
                    rv = False
                else:
                    lv = False

        return lv, rv

    def find_lanes(self, binary_warped):
        self.force_sliding_window_counter += 1
        force = (self.force_sw or
                 self.force_sliding_window_counter >= self.FORCE_SLIDING_WINDOW_INTERVAL or
                 self.low_confidence_counter >= self.LOW_CONF_THRESHOLD or
                 not self.detected or self.recovery_mode)
        if force:
            self.force_sliding_window_counter = 0

        has_fits = self.left_fit is not None and self.right_fit is not None

        if not self._sw_bootstrapped and not has_fits and self.use_prior:
            lx, ly, rx, ry = self.find_lanes_sliding_window(binary_warped)
            self.viz_method = "bootstrap"
            self._sw_bootstrapped = True
        elif self.use_prior and not self.force_sw and has_fits:
            lx, ly, rx, ry = self.find_lanes_from_prior(binary_warped)
            if len(lx) < self.prior_min_points or len(rx) < self.prior_min_points:
                if self.use_sliding_window:
                    lx, ly, rx, ry = self.find_lanes_sliding_window(binary_warped)
                    self.viz_method = "sliding_window"
                else:
                    self.viz_method = "prior_failed"
            else:
                self.viz_method = "prior"
        elif self.use_sliding_window:
            lx, ly, rx, ry = self.find_lanes_sliding_window(binary_warped)
            self.viz_method = "sliding_window"
        else:
            self.viz_method = "none"
            lx, ly, rx, ry = np.array([]), np.array([]), np.array([]), np.array([])

        lf_new, cl = self.fit_polynomial(lx, ly)
        rf_new, cr = self.fit_polynomial(rx, ry)

        if self.detect_lane_jump(lf_new, rf_new):
            cl *= 0.3
            cr *= 0.3

        self.confidence_left = cl
        self.confidence_right = cr

        if cl < self.MIN_CONFIDENCE and cr < self.MIN_CONFIDENCE:
            self.low_confidence_counter += 1
            self.roi_expand_factor = min(self.MAX_ROI_EXPAND, self.roi_expand_factor + 0.1)
            if self.low_confidence_counter >= self.LOW_CONF_THRESHOLD:
                self.reset_tracker("Low confidence")
        else:
            self.low_confidence_counter = 0
            self.roi_expand_factor = max(1.0, self.roi_expand_factor - 0.05)

        lv, rv = self.validate_lanes(lf_new, rf_new)
        if not lv:
            lf_new = None
            cl = 0.0
        if not rv:
            rf_new = None
            cr = 0.0

        self.left_fit = self.smooth_fit(lf_new, self.left_fit_history, cl)
        self.right_fit = self.smooth_fit(rf_new, self.right_fit_history, cr)
        self._handle_missing_lanes()

        self.left_detected = self.left_fit is not None
        self.right_detected = self.right_fit is not None
        self.detected = self.left_detected or self.right_detected

        if self.left_detected and cl > 0.6:
            self.last_good_left_fit = self.left_fit.copy()
        if self.right_detected and cr > 0.6:
            self.last_good_right_fit = self.right_fit.copy()

        self.update_position_tracking()

        if self.detected:
            self._calculate_curvature()
            if self.recovery_mode:
                self.recovery_mode = False
                self.recovery_counter = 0

    def _handle_missing_lanes(self):
        if self.left_fit is not None and self.right_fit is None:
            self.right_fit = np.array([
                self.left_fit[0],
                self.left_fit[1],
                self.left_fit[2] + self.lane_width
            ])
        elif self.right_fit is not None and self.left_fit is None:
            self.left_fit = np.array([
                self.right_fit[0],
                self.right_fit[1],
                self.right_fit[2] - self.lane_width
            ])
        elif self.left_fit is None and self.right_fit is None:
            if self.last_good_left_fit is not None and self.blind_counter < 10:
                self.left_fit = self.last_good_left_fit
            if self.last_good_right_fit is not None and self.blind_counter < 10:
                self.right_fit = self.last_good_right_fit

    def _calculate_curvature(self):
        if self.left_fit is None and self.right_fit is None:
            self.curvature_radius = float('inf')
            self.curve_direction = 0
            return
        ye = self.image_height
        f = self.left_fit if self.left_fit is not None else self.right_fit
        A, B = f[0], f[1]
        if abs(A) < 1e-6:
            self.curvature_radius = float('inf')
            self.curve_direction = 0
        else:
            self.curvature_radius = abs((1 + (2 * A * ye + B)**2)**1.5 / (2 * A))
            self.curve_direction = 1 if A < -0.0001 else (-1 if A > 0.0001 else 0)

        self.last_steer_time = None
        self.filtered_derivative = 0.0
        self.integral_error = 0.0
        self.current_speed = 2.0

        # Optimal High-Speed Tuned Parameters
        self.kp_base = 0.90
        self.kd_base = 0.035
        self.ki_base = 0.0015
        self.k_heading = 0.50
        self.kc = 0.20
        self.d_filter_alpha = 0.30
        self.max_pixel_offset = 100.0

    def set_current_speed(self, speed):
        self.current_speed = max(1.0, float(speed))

    def calculate_steering(self, current_time_sec=None):
        if not self.detected:
            self.blind_counter += 1
            self.integral_error = 0.0  # Anti-windup reset on lane loss
            if not self.dr_mode:
                self.dr_mode = True
                self.dr_counter = 0
                self.dr_last_steer = self.last_good_steer
                self.dr_last_curvature = self.curvature_radius
                self.dr_last_curve_dir = self.curve_direction
                self.mode = "DEAD_RECKONING"

            self.dr_counter += 1
            decay = max(0.0, 1.0 - self.dr_counter / self.DEAD_RECKON_MAX_FRAMES)

            if self.dr_counter < self.DEAD_RECKON_MAX_FRAMES:
                if self.dr_last_curvature < 500 and self.dr_last_curve_dir != 0:
                    steer = self.dr_last_curve_dir * 0.35 * decay
                else:
                    steer = self.dr_last_steer * decay
                self.recovery_mode = True
                self.recovery_counter = self.dr_counter
                return float(np.clip(steer, -1.0, 1.0))
            else:
                self.dr_mode = False
                self.recovery_mode = False
                self.mode = "VISION_TRACKING"
                return 0.0

        if self.dr_mode:
            self.dr_mode = False
            self.dr_counter = 0
            self.mode = "VISION_TRACKING"

        self.blind_counter = 0
        self.recovery_mode = False
        if self.curvature_radius < 500:
            self.was_in_heavy_turn = True
            self.last_turn_direction = self.curve_direction

        # ── 1. Calculate Precise Time Delta (dt) ─────────────────────────────
        if current_time_sec is None:
            current_time_sec = time.time()
        if self.last_steer_time is None:
            dt = 1.0 / 30.0
        else:
            dt = current_time_sec - self.last_steer_time
            if dt <= 0.001 or dt > 0.5:
                dt = 1.0 / 30.0
        self.last_steer_time = current_time_sec

        # ── 2. Speed-Adaptive Lookahead Points ────────────────────────────────
        # Preview points (in pixels from bottom y=300 to top y=0)
        speed_factor = min(1.0, max(0.0, (self.current_speed - 1.0) / 5.0))
        near_ratio = 0.88 - 0.08 * speed_factor   # 0.88 -> 0.80
        mid_ratio  = 0.65 - 0.15 * speed_factor   # 0.65 -> 0.50
        far_ratio  = 0.45 - 0.20 * speed_factor   # 0.45 -> 0.25

        yn = int(self.image_height * near_ratio)
        ym = int(self.image_height * mid_ratio)
        yf = int(self.image_height * far_ratio)

        def lc(y):
            l = np.polyval(self.left_fit, y) if self.left_fit is not None else 0
            r = np.polyval(self.right_fit, y) if self.right_fit is not None else self.image_width
            return (l + r) / 2.0

        cn, cm, cf = lc(yn), lc(ym), lc(yf)
        cc = self.image_width / 2.0

        # ── 3. Consistent Multi-Point Error ──────────────────────────────────
        en = cn - cc
        em = cm - cc
        ef = cf - cc

        # Blend near (tracking), mid (stability), and far (curve anticipation)
        w_near = 0.45 - 0.15 * speed_factor   # 0.45 -> 0.30
        w_mid  = 0.35 + 0.05 * speed_factor   # 0.35 -> 0.40
        w_far  = 0.20 + 0.10 * speed_factor   # 0.20 -> 0.30

        err = w_near * en + w_mid * em + w_far * ef

        # ── 4. Robust PID Control ────────────────────────────────────────────
        effective_kp = 1.05
        p_term = err * effective_kp

        raw_derivative = (err - self.prev_error) / dt
        self.filtered_derivative = (0.35 * raw_derivative + 0.65 * self.filtered_derivative)
        d_term = self.filtered_derivative * 0.035

        # Anti-windup Integral Term
        if abs(err) < 60.0:
            self.integral_error = max(-40.0, min(40.0, self.integral_error + err * dt))
        else:
            self.integral_error *= 0.5
        i_term = self.integral_error * 0.002

        # ── 5. Curvature Feedforward ──────────────────────────────────────────
        cs = 0.0
        if self.curvature_radius < 1200:
            curv_intensity = (600.0 / max(self.curvature_radius, 40.0)) ** 1.15
            cs = self.curve_direction * 0.35 * min(curv_intensity, 2.5)

        # ── 6. Combined Normalized Steering ──────────────────────────────────
        raw_steer = (p_term + d_term + i_term + cs) / 100.0
        steer = float(np.clip(raw_steer, -1.0, 1.0))

        self.prev_error = err
        self.last_good_steer = steer
        return steer

    def is_in_recovery(self):
        return self.recovery_mode or not self.detected or self.dr_mode


class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detection_node')

        rt_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.subscription = self.create_subscription(
            Image, '/filtered_image', self.listener_callback, rt_qos)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, rt_qos)
        self.speed_sub = self.create_subscription(
            Int8, '/cmd_vel', self.speed_callback, rt_qos)
        self.steer_pub = self.create_publisher(Float32, '/steering_value', rt_qos)
        self.status_pub = self.create_publisher(String, '/lane_status', rt_qos)
        self.debug_pub = self.create_publisher(Image, '/debug_image', rt_qos)

        self.br = CvBridge()
        self.tracker = None
        self.frame_count = 0
        self.current_speed = 2.0

        self.declare_parameter('kp', 0.90)
        self.declare_parameter('kd', 0.035)
        self.declare_parameter('ki', 0.0015)
        self.declare_parameter('kc', 0.20)
        self.declare_parameter('nwindows', 6)
        self.declare_parameter('minpix', 15)
        self.declare_parameter('base_margin', 60)
        self.declare_parameter('force_sw_interval', 10)
        self.declare_parameter('max_empty', 4)
        self.declare_parameter('roi_expand_max', 1.5)
        self.declare_parameter('min_points', 50)
        self.declare_parameter('min_confidence', 0.35)
        self.declare_parameter('low_conf_threshold', 2)
        self.declare_parameter('prior_margin', 50)
        self.declare_parameter('prior_min_points', 150)
        self.declare_parameter('max_jump_pixels', 80)
        self.declare_parameter('max_curvature_a', 0.005)
        self.declare_parameter('max_recovery_frames', 30)
        self.declare_parameter('max_pixel_offset', 100)
        self.declare_parameter('lane_width_ratio', 0.75)
        self.declare_parameter('show_windows', True)
        self.declare_parameter('show_histogram', True)
        self.declare_parameter('show_pixels', True)
        self.declare_parameter('show_fill', True)
        self.declare_parameter('use_prior', True)
        self.declare_parameter('use_sliding_window', False)
        self.declare_parameter('force_sw', False)
        self.declare_parameter('window_expand_per_empty', 15)
        self.declare_parameter('max_margin_base', 120)
        self.declare_parameter('preview_near_ratio', 0.85)
        self.declare_parameter('preview_far_ratio', 0.65)
        self.declare_parameter('dead_reckon_max_frames', 40)

        self.show_windows = True
        self.show_histogram = True
        self.show_pixels = True
        self.show_fill = True

        self.add_on_set_parameters_callback(self._param_callback)
        self.get_logger().info('LaneDetectionNode ready for simulator')

    def odom_callback(self, msg):
        if self.tracker is not None:
            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation
            theta = math.atan2(2.0 * (ori.w * ori.z + ori.x * ori.y),
                               1.0 - 2.0 * (ori.y * ori.y + ori.z * ori.z))
            self.tracker.update_odom(pos.x, pos.y, theta)

    def speed_callback(self, msg: Int8):
        self.current_speed = float(msg.data)
        if self.tracker is not None:
            self.tracker.set_current_speed(self.current_speed)

    def _load_params(self):
        p = self.get_parameter
        t = self.tracker
        t.kp_base = p('kp').value
        t.kd_base = p('kd').value
        t.ki_base = p('ki').value
        t.kc = p('kc').value
        t.nwindows = p('nwindows').value
        t.minpix = p('minpix').value
        t.base_margin = p('base_margin').value
        t.FORCE_SLIDING_WINDOW_INTERVAL = p('force_sw_interval').value
        t.max_empty = p('max_empty').value
        t.MAX_ROI_EXPAND = p('roi_expand_max').value
        t.MIN_POINTS = p('min_points').value
        t.MIN_CONFIDENCE = p('min_confidence').value
        t.LOW_CONF_THRESHOLD = p('low_conf_threshold').value
        t.prior_margin = p('prior_margin').value
        t.prior_min_points = p('prior_min_points').value
        t.MAX_JUMP_PIXELS = p('max_jump_pixels').value
        t.max_curvature_a = p('max_curvature_a').value
        t.MAX_RECOVERY_FRAMES = p('max_recovery_frames').value
        t.max_pixel_offset = p('max_pixel_offset').value
        t.lane_width = int(t.image_width * p('lane_width_ratio').value)
        t.use_prior = p('use_prior').value
        t.use_sliding_window = p('use_sliding_window').value
        t.force_sw = p('force_sw').value
        t.window_expand_per_empty = p('window_expand_per_empty').value
        t.max_margin_base = p('max_margin_base').value
        t.PREVIEW_NEAR_RATIO = p('preview_near_ratio').value
        t.PREVIEW_FAR_RATIO = p('preview_far_ratio').value
        t.DEAD_RECKON_MAX_FRAMES = p('dead_reckon_max_frames').value
        self.show_windows = p('show_windows').value
        self.show_histogram = p('show_histogram').value
        self.show_pixels = p('show_pixels').value
        self.show_fill = p('show_fill').value

    def _param_callback(self, params):
        tracker_map = {
            'kp': 'kp_base', 'kd': 'kd_base', 'ki': 'ki_base', 'kc': 'kc',
            'base_margin': 'base_margin', 'minpix': 'minpix',
            'nwindows': 'nwindows', 'force_sw_interval': 'FORCE_SLIDING_WINDOW_INTERVAL',
            'max_empty': 'max_empty', 'roi_expand_max': 'MAX_ROI_EXPAND',
            'min_points': 'MIN_POINTS', 'min_confidence': 'MIN_CONFIDENCE',
            'low_conf_threshold': 'LOW_CONF_THRESHOLD',
            'prior_margin': 'prior_margin', 'prior_min_points': 'prior_min_points',
            'max_jump_pixels': 'MAX_JUMP_PIXELS',
            'max_curvature_a': 'max_curvature_a',
            'max_recovery_frames': 'MAX_RECOVERY_FRAMES',
            'max_pixel_offset': 'max_pixel_offset',
            'use_prior': 'use_prior',
            'use_sliding_window': 'use_sliding_window',
            'force_sw': 'force_sw',
            'window_expand_per_empty': 'window_expand_per_empty',
            'max_margin_base': 'max_margin_base',
            'preview_near_ratio': 'PREVIEW_NEAR_RATIO',
            'preview_far_ratio': 'PREVIEW_FAR_RATIO',
            'dead_reckon_max_frames': 'DEAD_RECKON_MAX_FRAMES',
        }
        for param in params:
            if param.name in tracker_map and self.tracker is not None:
                setattr(self.tracker, tracker_map[param.name], param.value)
                self.get_logger().info(f'{param.name} = {param.value}')
            elif param.name == 'lane_width_ratio' and self.tracker is not None:
                self.tracker.lane_width = int(self.tracker.image_width * param.value)
                self.get_logger().info(f'lane_width = {self.tracker.lane_width}')
            elif param.name in ('show_windows', 'show_histogram', 'show_pixels', 'show_fill'):
                setattr(self, param.name, param.value)
                self.get_logger().info(f'{param.name} = {param.value}')
        return SetParametersResult(successful=True)

    def draw_debug_hud(self, binary, steer):
        t = self.tracker
        vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        h, w = vis.shape[:2]
        cx = w // 2

        cv2.line(vis, (cx, 0), (cx, h), (255, 255, 0), 1)

        if self.show_histogram and t.viz_histogram is not None:
            hist = t.viz_histogram
            hist_h = int(h * 0.25)
            hist_w = len(hist)
            if hist.max() > 0:
                hist_normalized = (hist / hist.max() * hist_h).astype(int)
                for i in range(0, hist_w, 3):
                    bar_h = hist_normalized[i]
                    if bar_h > 0:
                        x = int(i * w / hist_w)
                        cv2.line(vis, (x, h), (x, h - bar_h), (180, 180, 0), 1)

        if self.show_fill and t.left_fit is not None and t.right_fit is not None:
            y = np.linspace(0, h, 100)
            lx = np.polyval(t.left_fit, y).astype(int)
            rx = np.polyval(t.right_fit, y).astype(int)
            fill_pts = np.column_stack([
                np.concatenate([lx, rx[::-1]]),
                np.concatenate([y.astype(int), y[::-1].astype(int)])
            ]).astype(np.int32)
            overlay = vis.copy()
            cv2.fillPoly(overlay, [fill_pts], (0, 100, 50))
            cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)

        if self.show_windows:
            for (x_lo, x_hi, y_lo, y_hi) in t.viz_left_windows:
                cv2.rectangle(vis, (x_lo, y_lo), (x_hi, y_hi), (0, 200, 0), 1)
            for (x_lo, x_hi, y_lo, y_hi) in t.viz_right_windows:
                cv2.rectangle(vis, (x_lo, y_lo), (x_hi, y_hi), (0, 0, 200), 1)

        if t.left_fit is not None:
            y = np.linspace(0, h, 100)
            x = np.polyval(t.left_fit, y).astype(int)
            pts = np.column_stack((x, y)).astype(np.int32)
            cv2.polylines(vis, [pts], False, (0, 255, 0), 2)

        if t.right_fit is not None:
            y = np.linspace(0, h, 100)
            x = np.polyval(t.right_fit, y).astype(int)
            pts = np.column_stack((x, y)).astype(np.int32)
            cv2.polylines(vis, [pts], False, (0, 0, 255), 2)

        if t.left_fit is not None and t.right_fit is not None:
            y = np.linspace(0, h, 100)
            lx = np.polyval(t.left_fit, y)
            rx = np.polyval(t.right_fit, y)
            mid = ((lx + rx) / 2).astype(int)
            pts = np.column_stack((mid, y.astype(int))).astype(np.int32)
            cv2.polylines(vis, [pts], False, (255, 0, 255), 2)

        if self.show_pixels:
            lx, ly = t.viz_left_pixels
            if len(lx) > 0:
                for px, py in zip(lx[::4], ly[::4]):
                    cv2.circle(vis, (int(px), int(py)), 1, (0, 255, 0), -1)
            rx, ry = t.viz_right_pixels
            if len(rx) > 0:
                for px, py in zip(rx[::4], ry[::4]):
                    cv2.circle(vis, (int(px), int(py)), 1, (0, 0, 255), -1)

        curv_str = f"{t.curvature_radius:.0f}px" if t.curvature_radius < 1e6 else "INF"
        lines = [
            f"STEER: {steer:+.3f}",
            f"L: {'OK' if t.left_detected else 'X'} CL:{t.confidence_left:.2f}",
            f"R: {'OK' if t.right_detected else 'X'} CL:{t.confidence_right:.2f}",
            f"CURV: {curv_str}",
            f"MODE: {t.mode}",
            f"ROI: {t.roi_expand_factor:.2f}",
            f"KP:{t.kp:.1f} KD:{t.kd:.2f} KC:{t.kc:.2f}",
            f"METHOD: {t.viz_method.upper()}",
        ]

        y0 = 20
        for i, line in enumerate(lines):
            cv2.putText(vis, line, (10, y0 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        if t.dr_mode:
            badge_color = (0, 0, 200)
            badge_text = "DEAD_RECKON"
        elif t.recovery_mode:
            badge_color = (0, 0, 200)
            badge_text = "RECOVERY"
        elif t.force_sw:
            badge_color = (0, 200, 200)
            badge_text = "FORCE_SW"
        elif t.viz_method == "prior_failed":
            badge_color = (0, 165, 255)
            badge_text = "PRIOR_FAIL"
        elif t.viz_method == "none":
            badge_color = (80, 80, 80)
            badge_text = "DISABLED"
        elif t.detected:
            badge_color = (0, 200, 0)
            badge_text = t.viz_method.upper()
        else:
            badge_color = (0, 0, 200)
            badge_text = "LOST"

        badge_x = w - 130
        cv2.rectangle(vis, (badge_x, y0 - 5), (badge_x + 115, y0 + 16), badge_color, -1)
        cv2.putText(vis, badge_text, (badge_x + 5, y0 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        bar_x = w - 50
        bar_top = 40
        bar_bot = h - 20
        bar_h = bar_bot - bar_top
        cv2.rectangle(vis, (bar_x, bar_top), (bar_x + 16, bar_bot), (100, 100, 100), -1)
        cv2.line(vis, (bar_x, bar_top + bar_h // 2), (bar_x + 16, bar_top + bar_h // 2), (255, 255, 0), 1)
        steer_y = int(bar_top + bar_h / 2 - steer * (bar_h / 2))
        steer_y = max(bar_top, min(bar_bot, steer_y))
        color = (0, 255, 0) if abs(steer) < 0.3 else (0, 255, 255) if abs(steer) < 0.6 else (0, 0, 255)
        cv2.circle(vis, (bar_x + 8, steer_y), 6, color, -1)
        cv2.putText(vis, "L", (bar_x - 10, bar_top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        cv2.putText(vis, "R", (bar_x - 10, bar_bot + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        return vis

    def listener_callback(self, msg):
        try:
            binary = self.br.imgmsg_to_cv2(msg, desired_encoding='mono8')
            if self.tracker is None:
                h, w = binary.shape[:2]
                self.tracker = LaneTracker(w, h)
                self._load_params()
                self.get_logger().info(f'LaneTracker initialized: {w}x{h}')

            self.tracker.find_lanes(binary)
            t_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 if msg.header.stamp.sec > 0 else time.time()
            steer = self.tracker.calculate_steering(current_time_sec=t_sec)

            sm = Float32()
            sm.data = float(steer)
            self.steer_pub.publish(sm)

            status = json.dumps({
                "recovering": self.tracker.is_in_recovery(),
                "curvature": self.tracker.curvature_radius if self.tracker.curvature_radius != float('inf') else -1.0,
                "detected": self.tracker.detected,
                "mode": self.tracker.mode
            })
            st = String()
            st.data = status
            self.status_pub.publish(st)

            self.frame_count += 1
            if self.frame_count % 3 == 0:
                debug_img = self.draw_debug_hud(binary, steer)
                debug_msg = self.br.cv2_to_imgmsg(debug_img, encoding='bgr8')
                debug_msg.header = msg.header
                self.debug_pub.publish(debug_msg)

        except Exception as e:
            self.get_logger().error(f'LaneDetectionNode Error: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

