#!/usr/bin/env python3
"""
ipm_transformer.py
Inverse Perspective Mapping (IPM) & Metric Ground Space Transformer for AVIS Engine.

Converts front camera pixels (u, v) in 512x512 image into physical ground metric coordinates:
- X: Forward distance from rear axle/bumper (0 to 50 meters)
- Y: Lateral offset from vehicle center line (-7 to +7 meters, positive = right)

Provides:
1. Bidirectional Homography projection (pixel <-> ground meters).
2. Parametric Bézier metric curve unprojection.
3. Top-down radar-style HUD Mini-Map overlay on the camera debug image.
"""

import math
from typing import Tuple, Optional, List
import cv2
import numpy as np


class IPMTransformer:
    def __init__(self, image_width: int = 512, image_height: int = 512):
        self.w = image_width
        self.h = image_height

        # Calibrated 4-point correspondence for AvisEngine front camera
        # Near horizon: y=435, width=435px (lane width 3.5m) -> X=1.8m, Y in [-1.75, +1.75]
        # Mid/Far horizon: y=215, width=120px -> X=25.0m, Y in [-1.75, +1.75]
        src_pts = np.float32([
            [38.5,  435.0],   # Near Left
            [473.5, 435.0],   # Near Right
            [316.0, 215.0],   # Far Right (25m)
            [196.0, 215.0]    # Far Left (25m)
        ])

        dst_pts = np.float32([
            [1.8,  -1.75],    # X=1.8m,  Y=-1.75m
            [1.8,   1.75],    # X=1.8m,  Y=+1.75m
            [25.0,  1.75],    # X=25.0m, Y=+1.75m
            [25.0, -1.75]     # X=25.0m, Y=-1.75m
        ])

        # Perspective Homography matrices
        self.H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.H_inv = np.linalg.inv(self.H)

    def pixel_to_metric(self, u: float, v: float) -> Tuple[float, float]:
        """
        Transforms a camera pixel coordinate (u, v) to ground metric coordinate (X, Y) in meters.
        X: forward distance [0, 50m]
        Y: lateral offset [-7m, +7m] (positive = right)
        """
        p = np.array([u, v, 1.0], dtype=np.float64)
        res = self.H @ p
        w = res[2]
        if abs(w) < 1e-6:
            return 0.0, 0.0
        return float(res[0] / w), float(res[1] / w)

    def pixel_to_metric_batch(self, pts_uv: np.ndarray) -> np.ndarray:
        """
        Batch transformation from pixel array (N, 2) to metric array (N, 2) [X_m, Y_m].
        """
        if len(pts_uv) == 0:
            return np.empty((0, 2), dtype=np.float64)
        pts = pts_uv.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(pts, self.H)
        return transformed.reshape(-1, 2).astype(np.float64)

    def metric_to_pixel(self, x_m: float, y_m: float) -> Tuple[float, float]:
        """
        Transforms ground metric coordinate (X, Y) back to camera pixel (u, v).
        """
        p = np.array([x_m, y_m, 1.0], dtype=np.float64)
        res = self.H_inv @ p
        w = res[2]
        if abs(w) < 1e-6:
            return 256.0, 435.0
        return float(res[0] / w), float(res[1] / w)

    def metric_to_pixel_batch(self, pts_xy: np.ndarray) -> np.ndarray:
        """
        Batch transformation from metric array (N, 2) [X_m, Y_m] to pixel array (N, 2) [u, v].
        """
        if len(pts_xy) == 0:
            return np.empty((0, 2), dtype=np.float64)
        pts = pts_xy.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(pts, self.H_inv)
        return transformed.reshape(-1, 2).astype(np.float64)

    def project_bezier_to_metric(self, bezier_curve, num_samples: int = 35) -> np.ndarray:
        """
        Samples a CubicBezier in image space and unprojects it into a continuous metric path [X_m, Y_m].
        Returns array of shape (num_samples, 2), sorted by increasing forward distance X_m.
        """
        ts = np.linspace(0.0, 1.0, num_samples)
        pixel_pts = bezier_curve.eval_multi(ts)  # (N, 2) [u, v]
        metric_pts = self.pixel_to_metric_batch(pixel_pts)

        # Ensure sorted by forward distance X (from vehicle bumper forward to 50m)
        if len(metric_pts) > 1 and metric_pts[0, 0] > metric_pts[-1, 0]:
            metric_pts = metric_pts[::-1]

        return metric_pts

    def render_hud_minimap(self,
                           frame: np.ndarray,
                           center_metric: Optional[np.ndarray],
                           left_metric: Optional[np.ndarray] = None,
                           right_metric: Optional[np.ndarray] = None,
                           target_pt_metric: Optional[Tuple[float, float]] = None,
                           origin_offset: Tuple[int, int] = (360, 10),
                           map_size: Tuple[int, int] = (140, 230)) -> np.ndarray:
        """
        Renders an inset top-down radar-style HUD Mini-Map directly on the camera frame.
        - Red point = point-robot vehicle at (0, 0)
        - Cyan line = 50m preview center path
        - Yellow/White lines = left and right lane boundaries in ground meters
        - Range rings at 10m, 20m, 30m, 40m
        """
        vis = frame.copy()
        ox, oy = origin_offset
        mw, mh = map_size

        if oy + mh > vis.shape[0] or ox + mw > vis.shape[1]:
            return vis

        # 1. Semi-transparent background
        sub_img = vis[oy:oy + mh, ox:ox + mw]
        dark_bg = np.zeros_like(sub_img)
        dark_bg[:] = (20, 20, 20)
        cv2.addWeighted(sub_img, 0.25, dark_bg, 0.75, 0, sub_img)

        # Border
        cv2.rectangle(vis, (ox, oy), (ox + mw, oy + mh), (100, 100, 100), 1)
        cv2.putText(vis, "BEV Mini-Map (50m)", (ox + 8, oy + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)

        # Scaling: X in [0, 48m] -> py in [mh-25, 25] (inverted)
        #          Y in [-6m, +6m] -> px in [15, mw-15] (center at mw/2)
        cx_map = ox + int(mw * 0.5)
        bot_y = oy + mh - 22
        top_y = oy + 25

        def to_map(xm: float, ym: float) -> Tuple[int, int]:
            px = int(cx_map + (ym / 5.5) * (mw * 0.45))
            py = int(bot_y - (xm / 45.0) * (bot_y - top_y))
            return max(ox + 2, min(ox + mw - 2, px)), max(oy + 2, min(oy + mh - 2, py))

        # 2. Distance range markers (10m, 20m, 30m, 40m)
        for dist_m in [10, 20, 30, 40]:
            _, ry = to_map(dist_m, 0.0)
            cv2.line(vis, (ox + 10, ry), (ox + mw - 10, ry), (45, 45, 45), 1)
            cv2.putText(vis, f"{dist_m}m", (ox + mw - 28, ry - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (120, 120, 120), 1, cv2.LINE_AA)

        # 3. Draw Left Lane Boundary (Yellow)
        if left_metric is not None and len(left_metric) > 1:
            pts_l = [to_map(p[0], p[1]) for p in left_metric if 0.0 <= p[0] <= 45.0]
            for i in range(len(pts_l) - 1):
                cv2.line(vis, pts_l[i], pts_l[i + 1], (0, 220, 220), 1, cv2.LINE_AA)

        # 4. Draw Right Lane Boundary (White)
        if right_metric is not None and len(right_metric) > 1:
            pts_r = [to_map(p[0], p[1]) for p in right_metric if 0.0 <= p[0] <= 45.0]
            for i in range(len(pts_r) - 1):
                cv2.line(vis, pts_r[i], pts_r[i + 1], (230, 230, 230), 1, cv2.LINE_AA)

        # 5. Draw Center Planned Trajectory (Bright Cyan)
        if center_metric is not None and len(center_metric) > 1:
            pts_c = [to_map(p[0], p[1]) for p in center_metric if 0.0 <= p[0] <= 45.0]
            for i in range(len(pts_c) - 1):
                cv2.line(vis, pts_c[i], pts_c[i + 1], (255, 255, 0), 2, cv2.LINE_AA)

        # 6. Draw Lookahead Target Point on Path
        if target_pt_metric is not None:
            tx, ty = to_map(target_pt_metric[0], target_pt_metric[1])
            cv2.circle(vis, (tx, ty), 4, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(vis, (tx, ty), 6, (0, 255, 0), 1, cv2.LINE_AA)

        # 7. Draw Point-Robot Vehicle at (0, 0)
        rx, ry = to_map(0.0, 0.0)
        cv2.circle(vis, (rx, ry), 5, (0, 0, 255), -1, cv2.LINE_AA)
        # Small forward heading triangle
        tip = (rx, ry - 7)
        left_corner = (rx - 4, ry + 3)
        right_corner = (rx + 4, ry + 3)
        cv2.drawContours(vis, [np.array([tip, left_corner, right_corner])], 0, (0, 255, 255), -1)

        return vis
