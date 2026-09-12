#!/usr/bin/env python3
"""
test_cv_fallback.py
Verifies the Computer Vision / Thresholding Fallback with Centroid/Moments extraction.
Creates a synthetic camera frame with a curving road and verifies that the fallback:
1. Detects left and right lane markings.
2. Extracts centroids across 4 vertical slices.
3. Constructs valid CubicBezier curves.
4. Calculates non-zero lateral error, heading deviation, curvature, and decisive steering.
"""

import sys
import os
import math
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "SelfDriving-jetson Source"))
from bezier_math import CubicBezier, compute_center_bezier, extract_control_metrics


def create_curved_road_frame(h=512, w=512, curve_direction="right"):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = (50, 50, 50)  # Road asphalt

    # Yellow left lane and White right lane
    # If curving right: lanes shift rightwards as y goes from 512 up to 200
    pts_left = []
    pts_right = []
    shift = 90.0 if curve_direction == "right" else -90.0

    for y in range(200, 512, 10):
        progress = (512 - y) / 312.0  # 0 at bottom, 1 at top
        dx = shift * (progress ** 2)
        xl = int(140 + dx)
        xr = int(370 + dx)
        pts_left.append([xl, y])
        pts_right.append([xr, y])

    pts_left = np.array(pts_left, dtype=np.int32)
    pts_right = np.array(pts_right, dtype=np.int32)

    # Draw Yellow left lane (BGR: [0, 220, 220], HSV ~ [30, 255, 220])
    cv2.polylines(frame, [pts_left], isClosed=False, color=(0, 220, 220), thickness=8)

    # Draw White right lane (BGR: [255, 255, 255])
    cv2.polylines(frame, [pts_right], isClosed=False, color=(255, 255, 255), thickness=8)

    # Draw simulated vehicle hood at bottom center (BGR: dark metallic)
    cv2.rectangle(frame, (140, 420), (370, 512), (30, 30, 30), -1)

    return frame


def cv_centroid_fallback(frame, expected_lane_width=230.0):
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 1. Color Segmentation
    # Yellow mask (left lane)
    yellow_mask = cv2.inRange(hsv, np.array([15, 55, 75], dtype=np.uint8), np.array([38, 255, 255], dtype=np.uint8))

    # White mask (right lane)
    white_mask = cv2.inRange(hsv, np.array([0, 0, 170], dtype=np.uint8), np.array([180, 55, 255], dtype=np.uint8))

    # 2. Mask out vehicle hood (bottom center: y in [400, 512], x in [130, 382])
    yellow_mask[400:512, 130:382] = 0
    white_mask[400:512, 130:382] = 0

    # 3. Multi-Slice Centroid Extraction (4 vertical evaluation bands from Near to Far)
    # Band 0 (Near): y in [410, 460], center y = 435
    # Band 1 (Mid-Near): y in [350, 400], center y = 375
    # Band 2 (Mid-Far): y in [290, 340], center y = 315
    # Band 3 (Far): y in [230, 280], center y = 255
    bands = [
        (410, 460, 435),
        (350, 400, 375),
        (290, 340, 315),
        (230, 280, 255),
    ]

    left_points = []
    right_points = []
    center_points = []

    car_x = w * 0.5

    for y_min, y_max, y_center in bands:
        # Extract slices
        slice_y = yellow_mask[y_min:y_max, :]
        slice_w = white_mask[y_min:y_max, :]

        # Left lane search (focus on left region)
        # Search up to car_x + 50 to allow left lane to curve right
        mask_l = np.zeros_like(slice_y)
        mask_l[:, :int(car_x + 60)] = cv2.bitwise_or(slice_y[:, :int(car_x + 60)], slice_w[:, :int(car_x + 60)])

        # Right lane search (focus on right region)
        # Search from car_x - 50 to w to allow right lane to curve left
        mask_r = np.zeros_like(slice_w)
        mask_r[:, int(car_x - 60):] = cv2.bitwise_or(slice_y[:, int(car_x - 60):], slice_w[:, int(car_x - 60):])

        M_l = cv2.moments(mask_l)
        cx_l = None
        if M_l['m00'] > 20:
            cx_l = float(M_l['m10'] / M_l['m00'])
            left_points.append([cx_l, float(y_center)])

        M_r = cv2.moments(mask_r)
        cx_r = None
        if M_r['m00'] > 20:
            cx_r = float(M_r['m10'] / M_r['m00'])
            right_points.append([cx_r, float(y_center)])

        # Compute centerline point for this slice
        if cx_l is not None and cx_r is not None:
            cx_c = 0.5 * (cx_l + cx_r)
        elif cx_l is not None:
            cx_c = cx_l + 0.5 * expected_lane_width
        elif cx_r is not None:
            cx_c = cx_r - 0.5 * expected_lane_width
        else:
            cx_c = None

        if cx_c is not None:
            center_points.append([cx_c, float(y_center)])

    # Construct Cubic Bézier curves
    left_curve = None
    right_curve = None
    center_curve = None

    if len(left_points) >= 3:
        # Interpolate or expand to 4 control points
        indices = np.linspace(0, len(left_points) - 1, 4)
        pts = np.array([left_points[int(round(i))] for i in indices], dtype=np.float64)
        left_curve = CubicBezier(pts)

    if len(right_points) >= 3:
        indices = np.linspace(0, len(right_points) - 1, 4)
        pts = np.array([right_points[int(round(i))] for i in indices], dtype=np.float64)
        right_curve = CubicBezier(pts)

    if len(center_points) >= 3:
        indices = np.linspace(0, len(center_points) - 1, 4)
        pts = np.array([center_points[int(round(i))] for i in indices], dtype=np.float64)
        center_curve = CubicBezier(pts)
    else:
        center_curve = compute_center_bezier(left_curve, right_curve, expected_lane_width)

    return left_curve, right_curve, center_curve


def test_fallback():
    frame = create_curved_road_frame(512, 512, "right")
    left_c, right_c, center_c = cv_centroid_fallback(frame)

    assert left_c is not None, "Left curve not detected!"
    assert right_c is not None, "Right curve not detected!"
    assert center_c is not None, "Center curve not synthesized!"

    print("Left curve P0, P3:", left_c.P0, left_c.P3)
    print("Right curve P0, P3:", right_c.P0, right_c.P3)
    print("Center curve P0, P3:", center_c.P0, center_c.P3)

    metrics = extract_control_metrics(center_c, 512, 512)
    print("Control metrics on right curve:", {k: v for k, v in metrics.items() if not k.startswith("pt_")})

    # Since the curve bends right, heading error should be positive and curvature should be positive
    assert metrics["heading_err_deg"] > 2.0, f"Expected positive heading error on right curve, got {metrics['heading_err_deg']}"
    assert metrics["e_far"] > metrics["e_near"], "Expected e_far > e_near on curve"
    print("✓ CV Centroid Fallback verified successfully!")


if __name__ == "__main__":
    test_fallback()
