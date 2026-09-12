#!/usr/bin/env python3
"""
collect_bezier_dataset.py
Automated Dataset Recorder for BézierLaneNet in AvisEngine Simulator.

Collects real simulator camera frames while the vehicle drives autonomously,
computes ground-truth parametric cubic Bézier control points using verified
multi-horizon centroid tracking, and saves them in a format directly consumable
by BezierLaneNet training routines.
"""

import sys
import os
import time
import json
import socket
import base64
import argparse
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "SelfDriving-jetson Source"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "BezierLaneNet"))

from bezier_math import CubicBezier, compute_center_bezier, extract_control_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Collect dataset for BézierLaneNet from AvisEngine")
    parser.add_argument("--num_frames", type=int, default=300, help="Number of valid labeled frames to collect")
    parser.add_argument("--output_dir", type=str, default="dataset/avis_bezier", help="Output directory")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Simulator IP")
    parser.add_argument("--port", type=int, default=25001, help="Simulator Port")
    parser.add_argument("--speed", type=int, default=20, help="Vehicle speed throttle [15-30]")
    return parser.parse_args()


def detect_bezier_lanes_from_bgr(frame: np.ndarray):
    """
    Extracts left and right cubic Bézier curves from a simulator frame.
    Returns (left_bezier, right_bezier, center_bezier) or None.
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Yellow mask (left lane)
    yellow_mask = cv2.inRange(hsv, np.array([15, 55, 75], dtype=np.uint8), np.array([38, 255, 255], dtype=np.uint8))
    # White mask (right lane)
    white_mask = cv2.inRange(hsv, np.array([0, 0, 170], dtype=np.uint8), np.array([180, 55, 255], dtype=np.uint8))

    # Mask vehicle hood
    yellow_mask[400:512, 130:382] = 0
    white_mask[400:512, 130:382] = 0

    bands = [
        (410, 460, 435.0),
        (350, 400, 375.0),
        (290, 340, 315.0),
        (230, 280, 255.0),
    ]

    left_pts = []
    right_pts = []

    for y_min, y_max, y_center in bands:
        slice_y = yellow_mask[y_min:y_max, :]
        slice_w = white_mask[y_min:y_max, :]

        my = cv2.moments(slice_y)
        if my['m00'] > 40:
            left_pts.append(np.array([my['m10'] / my['m00'], y_center], dtype=np.float64))

        mw = cv2.moments(slice_w)
        if mw['m00'] > 40:
            right_pts.append(np.array([mw['m10'] / mw['m00'], y_center], dtype=np.float64))

    left_curve = None
    right_curve = None

    if len(left_pts) >= 3:
        if len(left_pts) == 3:
            p0, p1, p2 = left_pts
            pts = [p0, 0.67 * p0 + 0.33 * p1, 0.33 * p1 + 0.67 * p2, p2]
        else:
            pts = left_pts[:4]
        left_curve = CubicBezier(np.array(pts))

    if len(right_pts) >= 3:
        if len(right_pts) == 3:
            p0, p1, p2 = right_pts
            pts = [p0, 0.67 * p0 + 0.33 * p1, 0.33 * p1 + 0.67 * p2, p2]
        else:
            pts = right_pts[:4]
        right_curve = CubicBezier(np.array(pts))

    center_curve = None
    if left_curve is not None or right_curve is not None:
        center_curve = compute_center_bezier(left_curve, right_curve, 230.0)

    return left_curve, right_curve, center_curve


def main():
    args = parse_args()
    print("=" * 65)
    print("  BézierLaneNet Dataset Recorder for AvisEngine  ")
    print(f"  Target: {args.num_frames} frames -> {args.output_dir}")
    print("=" * 65)

    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    labels_file = os.path.join(args.output_dir, "labels.json")

    # Connect to simulator
    print(f"[INFO] Connecting to AvisEngine at {args.host}:{args.port}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5.0)
    try:
        sock.connect((args.host, args.port))
        print("[INFO] Connected to AvisEngine socket successfully!\n")
    except Exception as e:
        print(f"[ERROR] Could not connect to simulator: {e}")
        print("Please ensure AVISEngine is running and track is loaded.")
        return

    collected = 0
    records = []
    current_steer = 0
    kp, kd = 0.144, 3.0
    prev_err = 0.0

    try:
        while collected < args.num_frames:
            # Request frame & send current driving command
            cmd = f"Speed:{args.speed},Steering:{current_steer},ImageStatus:1,SensorStatus:1,GetSpeed:1,SensorAngle:30\n"
            sock.sendall(cmd.encode('utf-8'))

            # Read response until <EOF>
            buf = bytearray()
            while b'<EOF>' not in buf:
                chunk = sock.recv(16384)
                if not chunk:
                    break
                buf.extend(chunk)

            buf_str = buf.decode('utf-8', errors='ignore')
            if '<image>' not in buf_str or '</image>' not in buf_str:
                time.sleep(0.01)
                continue

            # Decode frame
            img_b64 = buf_str.split('<image>')[1].split('</image>')[0]
            raw_bytes = base64.b64decode(img_b64)
            frame = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            h, w = frame.shape[:2]

            # Detect Bézier lanes
            left_c, right_c, center_c = detect_bezier_lanes_from_bgr(frame)

            # Steer vehicle to keep driving automatically while collecting data
            if center_c is not None:
                metrics = extract_control_metrics(center_c, w, h, 0.85, 0.65, 0.45)
                err = metrics['blended_err']
                deriv = err - prev_err
                prev_err = err
                curv_ff = metrics.get('curvature', 0.0) * 0.8 * (h * 0.5)
                norm_steer = np.clip((kp * err + kd * deriv + curv_ff) / 100.0, -1.0, 1.0)
                current_steer = int(norm_steer * 75)  # Scale to AvisEngine [-100, 100]
            else:
                current_steer = 0

            # Only save if at least one lane is clearly detected
            if left_c is not None or right_c is not None:
                img_filename = f"frame_{collected:05d}.jpg"
                img_path = os.path.join(images_dir, img_filename)
                cv2.imwrite(img_path, frame)

                # Format control points scaled for BezierLaneNet input (820x295 standard)
                scale_x = 820.0 / w
                scale_y = 295.0 / h

                lane_ctrl_pts = []
                if left_c is not None:
                    pts_l = left_c.pts.copy()
                    pts_l[:, 0] *= scale_x
                    pts_l[:, 1] *= scale_y
                    lane_ctrl_pts.append({
                        "id": "left",
                        "control_points": pts_l.tolist()
                    })

                if right_c is not None:
                    pts_r = right_c.pts.copy()
                    pts_r[:, 0] *= scale_x
                    pts_r[:, 1] *= scale_y
                    lane_ctrl_pts.append({
                        "id": "right",
                        "control_points": pts_r.tolist()
                    })

                record = {
                    "image_file": img_filename,
                    "lanes_count": len(lane_ctrl_pts),
                    "lanes": lane_ctrl_pts,
                    "steering_applied": current_steer,
                    "timestamp": time.time()
                }
                records.append(record)
                collected += 1

                if collected % 25 == 0 or collected == args.num_frames:
                    print(f"[PROGRESS] Collected {collected}/{args.num_frames} frames (Steering: {current_steer:+d})")

    except KeyboardInterrupt:
        print("\n[INFO] Collection interrupted by user.")
    finally:
        try:
            # Stop vehicle
            sock.sendall(b"Speed:0,Steering:0,ImageStatus:0,SensorStatus:0,GetSpeed:0,SensorAngle:30\n")
            sock.close()
        except Exception:
            pass

    # Save JSON annotations
    with open(labels_file, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\n[DONE] Successfully saved {len(records)} labeled frames to {args.output_dir}!")
    print(f"[INFO] Annotations file: {labels_file}")


if __name__ == '__main__':
    main()
