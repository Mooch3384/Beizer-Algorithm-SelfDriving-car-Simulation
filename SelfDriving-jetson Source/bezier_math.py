#!/usr/bin/env python3
"""
bezier_math.py
Analytical Cubic Bézier Curve Geometry and Control Module.

Provides exact analytical formulations for:
- Cubic Bézier curve evaluation B(t)
- First derivative B'(t) (tangent / velocity vector)
- Second derivative B''(t) (acceleration / curvature vector)
- Heading angle psi(t) and heading error Delta psi
- Signed road curvature kappa(t) and curvature radius R(t)
- Fast lookahead root-finding (t for given y)
- Multi-lookahead lateral deviation extraction
"""

import math
import numpy as np
from typing import Tuple, List, Optional


class CubicBezier:
    """
    Parametric Cubic Bézier Curve:
    B(t) = (1-t)^3 * P0 + 3*(1-t)^2*t * P1 + 3*(1-t)*t^2 * P2 + t^3 * P3,  t in [0, 1]
    where P0, P1, P2, P3 are 2D control points: (x, y).
    """

    def __init__(self, control_points: np.ndarray):
        """
        Args:
            control_points: Array-like of shape (4, 2) representing [P0, P1, P2, P3].
        """
        pts = np.asarray(control_points, dtype=np.float64)
        if pts.shape != (4, 2):
            raise ValueError(f"Expected control points of shape (4, 2), got {pts.shape}")
        self.P0 = pts[0]
        self.P1 = pts[1]
        self.P2 = pts[2]
        self.P3 = pts[3]
        self.pts = pts

    def eval(self, t: float) -> np.ndarray:
        """
        Evaluates B(t) at scalar t in [0, 1].
        Returns: np.ndarray of shape (2,) [x, y].
        """
        t = np.clip(t, 0.0, 1.0)
        u = 1.0 - t
        u2 = u * u
        u3 = u2 * u
        t2 = t * t
        t3 = t2 * t

        b = (u3 * self.P0 +
             3.0 * u2 * t * self.P1 +
             3.0 * u * t2 * self.P2 +
             t3 * self.P3)
        return b

    def eval_multi(self, ts: np.ndarray) -> np.ndarray:
        """
        Evaluates B(t) at array of ts in [0, 1].
        Returns: np.ndarray of shape (N, 2).
        """
        ts = np.clip(np.asarray(ts, dtype=np.float64), 0.0, 1.0)
        u = 1.0 - ts
        u2 = u * u
        u3 = u2 * u
        t2 = ts * ts
        t3 = t2 * ts

        # Broadcasting (N, 1) * (2,) -> (N, 2)
        out = (u3[:, None] * self.P0 +
               (3.0 * u2 * ts)[:, None] * self.P1 +
               (3.0 * u * t2)[:, None] * self.P2 +
               t3[:, None] * self.P3)
        return out

    def eval_first_derivative(self, t: float) -> np.ndarray:
        """
        Analytical First Derivative B'(t):
        B'(t) = 3*(1-t)^2*(P1 - P0) + 6*(1-t)*t*(P2 - P1) + 3*t^2*(P3 - P2)
        Returns: np.ndarray of shape (2,) [x'(t), y'(t)].
        """
        t = np.clip(t, 0.0, 1.0)
        u = 1.0 - t

        d0 = self.P1 - self.P0
        d1 = self.P2 - self.P1
        d2 = self.P3 - self.P2

        d = 3.0 * (u * u * d0 + 2.0 * u * t * d1 + t * t * d2)
        return d

    def eval_second_derivative(self, t: float) -> np.ndarray:
        """
        Analytical Second Derivative B''(t):
        B''(t) = 6*(1-t)*(P2 - 2*P1 + P0) + 6*t*(P3 - 2*P2 + P1)
        Returns: np.ndarray of shape (2,) [x''(t), y''(t)].
        """
        t = np.clip(t, 0.0, 1.0)
        u = 1.0 - t

        dd0 = self.P2 - 2.0 * self.P1 + self.P0
        dd1 = self.P3 - 2.0 * self.P2 + self.P1

        dd = 6.0 * (u * dd0 + t * dd1)
        return dd

    def heading(self, t: float) -> float:
        """
        Computes the tangent heading angle psi(t) relative to vehicle forward axis.
        In image coordinates (x right, y down, vehicle moving up towards y=0):
        The forward direction is (0, -1).
        Tangent vector is B'(t) = (dx/dt, dy/dt).
        If parameter t increases moving forward (y decreases, dy < 0):
            heading = atan2(dx, -dy)
        If parameter t increases moving backward (y increases, dy > 0):
            heading = atan2(-dx, dy)
        Returns: heading angle in radians (positive = curving right, negative = curving left).
        """
        d = self.eval_first_derivative(t)
        dx, dy = d[0], d[1]

        # Determine direction of increasing t relative to vehicle forward (decreasing y)
        if (self.P3[1] - self.P0[1]) < 0:
            # t=0 near bottom, t=1 near top (forward)
            heading_rad = math.atan2(dx, -dy)
        else:
            # t=0 near top, t=1 near bottom
            heading_rad = math.atan2(-dx, dy)

        return heading_rad

    def curvature(self, t: float) -> float:
        """
        Computes analytical signed road curvature kappa(t):
        kappa = (x' * y'' - y' * x'') / ((x'^2 + y'^2)^(3/2))
        Returns: signed curvature in 1/pixel (or 1/meter).
        Positive = curving right, Negative = curving left (aligned with heading).
        """
        d = self.eval_first_derivative(t)
        dd = self.eval_second_derivative(t)

        dx, dy = d[0], d[1]
        ddx, ddy = dd[0], dd[1]

        denom = (dx * dx + dy * dy) ** 1.5
        if denom < 1e-9:
            return 0.0

        raw_kappa = (dx * ddy - dy * ddx) / denom

        # Align sign: forward motion in image is decreasing y (P3[1] < P0[1]).
        # Along forward direction, a rightward bend has dx > 0 and increasing heading,
        # which produces raw_kappa > 0.
        if (self.P3[1] - self.P0[1]) < 0:
            kappa = raw_kappa
        else:
            kappa = -raw_kappa

        return float(kappa)

    def radius_of_curvature(self, t: float) -> float:
        """
        Radius of curvature R = 1 / |kappa|.
        Returns: radius in pixels or meters (capped at 9999.0 for straight paths).
        """
        k = abs(self.curvature(t))
        if k < 1e-6:
            return 9999.0
        return float(min(9999.0, 1.0 / k))

    def find_t_for_y(self, y_target: float, num_iterations: int = 20) -> float:
        """
        Finds parameter t in [0, 1] such that B(t).y == y_target using binary search.
        Assumes y is monotonic along the lane section of interest.
        """
        y0 = self.P0[1]
        y3 = self.P3[1]

        low, high = 0.0, 1.0
        # If target outside range, clamp to closest boundary
        min_y = min(y0, y3)
        max_y = max(y0, y3)
        if y_target <= min_y:
            return 1.0 if y3 <= y0 else 0.0
        if y_target >= max_y:
            return 0.0 if y3 <= y0 else 1.0

        for _ in range(num_iterations):
            mid = 0.5 * (low + high)
            curr_y = self.eval(mid)[1]
            if (y3 >= y0 and curr_y < y_target) or (y3 < y0 and curr_y > y_target):
                low = mid
            else:
                high = mid

        return 0.5 * (low + high)

    def sample_points(self, num_points: int = 64) -> np.ndarray:
        """
        Generates array of (x, y) coordinates for visualization.
        Shape: (num_points, 2), dtype: np.int32 for cv2.polylines.
        """
        ts = np.linspace(0.0, 1.0, num_points)
        pts = self.eval_multi(ts)
        return np.int32(pts)


def get_perspective_lane_width(y: float, base_width_scale: float = 1.0) -> float:
    """
    Computes perspective-accurate lane width in pixels on AvisEngine 512x512 camera:
    Near horizon (y=435): ~435 px
    Mid horizon (y=345): ~300 px
    Far horizon (y=255): ~155 px
    """
    ratio = float(np.clip((y - 255.0) / (435.0 - 255.0), 0.0, 1.0))
    nominal_w = 155.0 + ratio * (435.0 - 155.0)
    return float(nominal_w * base_width_scale)


def compute_center_bezier(left_curve: Optional[CubicBezier],
                          right_curve: Optional[CubicBezier],
                          expected_lane_width: float = 240.0) -> Optional[CubicBezier]:
    """
    Synthesizes the target center cubic Bézier curve:
    1. If both left and right curves are present:
       P_center = (P_left + P_right) / 2
       (Exact analytical property: average of two cubic Béziers is an exact cubic Bézier).
    2. If only left curve is present:
       Shift each control point right by half of the perspective-accurate lane width at that y.
    3. If only right curve is present:
       Shift each control point left by half of the perspective-accurate lane width at that y.
    4. If neither:
       Returns None.
    """
    if left_curve is not None and right_curve is not None:
        center_pts = 0.5 * (left_curve.pts + right_curve.pts)
        return CubicBezier(center_pts)
    elif left_curve is not None:
        pts = left_curve.pts.copy()
        for i in range(len(pts)):
            w = get_perspective_lane_width(pts[i, 1])
            pts[i, 0] += 0.5 * w
        return CubicBezier(pts)
    elif right_curve is not None:
        pts = right_curve.pts.copy()
        for i in range(len(pts)):
            w = get_perspective_lane_width(pts[i, 1])
            pts[i, 0] -= 0.5 * w
        return CubicBezier(pts)
    return None


def extract_control_metrics(center_curve: CubicBezier,
                            image_width: int = 820,
                            image_height: int = 295,
                            lookahead_near_ratio: float = 0.85,
                            lookahead_mid_ratio: float = 0.65,
                            lookahead_far_ratio: float = 0.45,
                            car_x_offset: Optional[float] = None,
                            weights: Optional[Tuple[float, float, float]] = None) -> dict:
    """
    Calculates vehicle control metrics from center Bézier curve:
    - Lateral errors at Near, Mid, and Far lookahead horizons
    - Blended lateral error
    - Heading angle error (Delta psi)
    - Road curvature kappa and radius R
    """
    car_x = float(image_width * 0.5 if car_x_offset is None else car_x_offset)

    y_near = image_height * lookahead_near_ratio
    y_mid = image_height * lookahead_mid_ratio
    y_far = image_height * lookahead_far_ratio

    t_near = center_curve.find_t_for_y(y_near)
    t_mid = center_curve.find_t_for_y(y_mid)
    t_far = center_curve.find_t_for_y(y_far)

    pt_near = center_curve.eval(t_near)
    pt_mid = center_curve.eval(t_mid)
    pt_far = center_curve.eval(t_far)

    # Lateral deviations: positive means center path is to the right of car center
    e_near = float(pt_near[0] - car_x)
    e_mid = float(pt_mid[0] - car_x)
    e_far = float(pt_far[0] - car_x)

    # Blended multi-lookahead error
    if weights is not None:
        w_near, w_mid, w_far = weights
    else:
        w_near, w_mid, w_far = 0.40, 0.35, 0.25
    blended_err = w_near * e_near + w_mid * e_mid + w_far * e_far

    # Heading error at mid lookahead
    heading_err = center_curve.heading(t_mid)

    # Curvature at mid lookahead
    curv = center_curve.curvature(t_mid)
    radius = center_curve.radius_of_curvature(t_mid)

    return {
        "e_near": e_near,
        "e_mid": e_mid,
        "e_far": e_far,
        "blended_err": blended_err,
        "heading_err_rad": heading_err,
        "heading_err_deg": math.degrees(heading_err),
        "curvature": curv,
        "radius": radius,
        "pt_near": (int(pt_near[0]), int(pt_near[1])),
        "pt_mid": (int(pt_mid[0]), int(pt_mid[1])),
        "pt_far": (int(pt_far[0]), int(pt_far[1])),
        "t_near": t_near,
        "t_mid": t_mid,
        "t_far": t_far,
    }
