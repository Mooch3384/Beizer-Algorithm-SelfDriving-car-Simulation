#!/usr/bin/env python3
"""
test_bezier_math.py
Unit tests verifying the analytical formulations in bezier_math.py:
- Endpoint properties B(0) == P0, B(1) == P3
- Analytical first derivative vs finite differences
- Analytical second derivative vs finite differences
- Analytical curvature for known geometries
- Binary search root-finding accuracy
- Center curve derivation and multi-lookahead error calculations
"""

import sys
import os
import math
import numpy as np

# Ensure SelfDriving-jetson Source is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "SelfDriving-jetson Source"))
from bezier_math import CubicBezier, compute_center_bezier, extract_control_metrics


def test_endpoints():
    P0 = np.array([100.0, 295.0])
    P1 = np.array([120.0, 200.0])
    P2 = np.array([150.0, 100.0])
    P3 = np.array([200.0, 0.0])
    curve = CubicBezier(np.array([P0, P1, P2, P3]))

    b0 = curve.eval(0.0)
    b1 = curve.eval(1.0)
    assert np.allclose(b0, P0), f"B(0) {b0} != P0 {P0}"
    assert np.allclose(b1, P3), f"B(1) {b1} != P3 {P3}"
    print("✓ Endpoints test passed!")


def test_first_derivative():
    P0 = np.array([100.0, 295.0])
    P1 = np.array([130.0, 210.0])
    P2 = np.array([180.0, 110.0])
    P3 = np.array([240.0, 10.0])
    curve = CubicBezier(np.array([P0, P1, P2, P3]))

    eps = 1e-5
    for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
        d_analytical = curve.eval_first_derivative(t)
        d_numerical = (curve.eval(t + eps) - curve.eval(t - eps)) / (2.0 * eps)
        assert np.allclose(d_analytical, d_numerical, atol=1e-4), (
            f"Derivative mismatch at t={t}: analytical={d_analytical}, numerical={d_numerical}"
        )
    print("✓ First derivative test passed!")


def test_second_derivative():
    P0 = np.array([100.0, 295.0])
    P1 = np.array([130.0, 210.0])
    P2 = np.array([180.0, 110.0])
    P3 = np.array([240.0, 10.0])
    curve = CubicBezier(np.array([P0, P1, P2, P3]))

    eps = 1e-5
    for t in [0.2, 0.4, 0.6, 0.8]:
        dd_analytical = curve.eval_second_derivative(t)
        dd_numerical = (curve.eval_first_derivative(t + eps) - curve.eval_first_derivative(t - eps)) / (2.0 * eps)
        assert np.allclose(dd_analytical, dd_numerical, atol=1e-3), (
            f"Second derivative mismatch at t={t}: analytical={dd_analytical}, numerical={dd_numerical}"
        )
    print("✓ Second derivative test passed!")


def test_straight_line_curvature():
    # Straight line: curvature should be identically zero
    P0 = np.array([200.0, 300.0])
    P1 = np.array([200.0, 200.0])
    P2 = np.array([200.0, 100.0])
    P3 = np.array([200.0, 0.0])
    curve = CubicBezier(np.array([P0, P1, P2, P3]))

    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        k = curve.curvature(t)
        r = curve.radius_of_curvature(t)
        psi = curve.heading(t)
        assert abs(k) < 1e-6, f"Curvature on straight line expected 0, got {k}"
        assert r >= 9999.0, f"Radius on straight line expected >= 9999, got {r}"
        assert abs(psi) < 1e-6, f"Heading error on straight line expected 0, got {psi}"
    print("✓ Straight line curvature & heading test passed!")


def test_root_finding():
    P0 = np.array([100.0, 295.0])
    P1 = np.array([150.0, 200.0])
    P2 = np.array([220.0, 100.0])
    P3 = np.array([300.0, 10.0])
    curve = CubicBezier(np.array([P0, P1, P2, P3]))

    for target_y in [250.0, 180.0, 100.0, 50.0]:
        t_found = curve.find_t_for_y(target_y)
        pt = curve.eval(t_found)
        assert abs(pt[1] - target_y) < 0.1, f"Root finding error: expected y={target_y}, got {pt[1]}"
    print("✓ Lookahead root-finding test passed!")


def test_center_curve_and_metrics():
    # Left lane
    l_P0 = np.array([150.0, 295.0])
    l_P1 = np.array([170.0, 200.0])
    l_P2 = np.array([200.0, 100.0])
    l_P3 = np.array([250.0, 10.0])
    left_c = CubicBezier(np.array([l_P0, l_P1, l_P2, l_P3]))

    # Right lane (offset by 200 px)
    r_P0 = np.array([350.0, 295.0])
    r_P1 = np.array([370.0, 200.0])
    r_P2 = np.array([400.0, 100.0])
    r_P3 = np.array([450.0, 10.0])
    right_c = CubicBezier(np.array([r_P0, r_P1, r_P2, r_P3]))

    center_c = compute_center_bezier(left_c, right_c)
    assert center_c is not None
    assert np.allclose(center_c.P0, [250.0, 295.0])
    assert np.allclose(center_c.P3, [350.0, 10.0])

    metrics = extract_control_metrics(center_c, image_width=820, image_height=295, car_x_offset=250.0)
    assert abs(metrics["e_near"]) < 20.0
    print("Metrics output:", {k: v for k, v in metrics.items() if not k.startswith("pt_")})
    print("✓ Center curve synthesis & control metrics test passed!")


if __name__ == "__main__":
    test_endpoints()
    test_first_derivative()
    test_second_derivative()
    test_straight_line_curvature()
    test_root_finding()
    test_center_curve_and_metrics()
    print("\nALL BÉZIER MATHEMATICAL TESTS PASSED SUCCESSFULLY!")
