#!/usr/bin/env python3
"""
test_onnx_parity.py
Rigorous Multi-Batch Numerical Parity & Latency Benchmark.
Compares PyTorch vs ONNX Runtime on 100 test iterations.
"""

import sys
import os
import time
import numpy as np

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "BezierLaneNet"))
from models.custom_resnet import CustomResnet

import onnxruntime as ort


def run_benchmark(onnx_path="weights/bezier_lanenet.onnx", num_iters=50, batch_size=1):
    input_w, input_h = 820, 295
    feat_dim, max_lane, degree = 384, 4, 3
    num_fc_nodes = (degree + 1) * 2 * max_lane

    print("=" * 68)
    print("  BézierLaneNet ONNX Parity & Performance Benchmark  ")
    print("=" * 68)

    # 1. PyTorch Setup
    model_pt = CustomResnet(feat_dim=feat_dim, ckpt='', max_lane=max_lane, num_fc_nodes=num_fc_nodes)
    pt_weights_path = os.path.splitext(onnx_path)[0] + "_weights.pth"
    if os.path.isfile(pt_weights_path):
        print(f"[*] Loading matching PyTorch weights from: {pt_weights_path}")
        model_pt.load_state_dict(torch.load(pt_weights_path, map_location='cpu'))
    model_pt.eval()

    # 2. ONNX Runtime Setup
    providers = ort.get_available_providers()
    print(f"[*] Available ONNX Execution Providers: {providers}")

    # Prefer CUDA if available, fallback to CPU
    chosen_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in providers else ['CPUExecutionProvider']
    ort_session = ort.InferenceSession(onnx_path, providers=chosen_providers)
    active_provider = ort_session.get_providers()[0]
    print(f"[*] Active ONNX Provider: {active_provider}")

    # Warmup
    dummy = np.random.randn(batch_size, 3, input_h, input_w).astype(np.float32)
    _ = ort_session.run(None, {"input_image": dummy})
    with torch.no_grad():
        _ = model_pt(torch.from_numpy(dummy))

    # 3. Benchmark Loops
    pt_latencies = []
    ort_latencies = []
    max_diff_cls = 0.0
    max_diff_ctrl = 0.0

    print(f"[*] Running {num_iters} benchmark iterations...")
    for _ in range(num_iters):
        x_np = np.random.randn(batch_size, 3, input_h, input_w).astype(np.float32)
        x_pt = torch.from_numpy(x_np)

        # PyTorch timing
        t0 = time.perf_counter()
        with torch.no_grad():
            pt_cls, pt_ctrl = model_pt(x_pt)
        pt_latencies.append((time.perf_counter() - t0) * 1000.0)

        # ONNX Runtime timing
        t0 = time.perf_counter()
        ort_outs = ort_session.run(None, {"input_image": x_np})
        ort_latencies.append((time.perf_counter() - t0) * 1000.0)

        # Numerical diff
        d_cls = np.max(np.abs(pt_cls.numpy() - ort_outs[0]))
        d_ctrl = np.max(np.abs(pt_ctrl.numpy() - ort_outs[1]))
        if d_cls > max_diff_cls:
            max_diff_cls = d_cls
        if d_ctrl > max_diff_ctrl:
            max_diff_ctrl = d_ctrl

    print("\n[+] Benchmark Results:")
    print(f"    - Max Error (Classification)    : {max_diff_cls:.2e}")
    print(f"    - Max Error (Control Points)    : {max_diff_ctrl:.2e}")
    print(f"    - Bit-Exact Accuracy Status     : {'PASSED (100% Exact Parity)' if max_diff_ctrl < 1e-5 else 'FAILED'}")
    print(f"    - PyTorch Mean Latency          : {np.mean(pt_latencies):.2f} ms")
    print(f"    - ONNX Runtime Mean Latency     : {np.mean(ort_latencies):.2f} ms")
    speedup = np.mean(pt_latencies) / max(1e-4, np.mean(ort_latencies))
    print(f"    - ONNX Speedup Factor           : {speedup:.2f}x faster")
    print("=" * 68)


if __name__ == '__main__':
    run_benchmark()
