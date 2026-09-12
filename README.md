# Autonomous Driving Simulation via Bézier Curve Path Synthesis & BézierLaneNet

[![ROS 2](https://img.shields.io/badge/ROS%202-Jazzy%20%7C%20Humble-22314E.svg?logo=ros&logoColor=white)](https://docs.ros.org/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

An end-to-end, high-performance autonomous vehicle control and perception pipeline deployed on the **AVIS Engine Simulator**. The pipeline uses **parametric cubic Bézier curves** for continuous road geometry representation, real-time lane tracking, analytical curvature feedforward, and low-latency lateral steering control.

---

## 📑 Table of Contents
- [Architecture Overview](#-architecture-overview)
- [Key Innovations & Control Formulation](#-key-innovations--control-formulation)
- [Repository Structure](#-repository-structure)
- [Prerequisites & Installation](#-prerequisites--installation)
- [Quick Start](#-quick-start)
- [Simulator Configuration](#-simulator-configuration)
- [Model Training & Dataset Tools](#-model-training--dataset-tools)
- [Automated Testing & Verification](#-automated-testing--verification)
- [License](#-license)

---

## 🚗 Architecture Overview

The system runs a distributed **ROS 2 multi-threaded node architecture**, communicating directly with the AVIS Engine simulator over a high-throughput binary TCP socket interface:

```
+-----------------------------------------------------------------------------------+
|                              AVIS Engine Simulator                                |
|  - Unity 3D Physics Engine                                                        |
|  - Camera Stream (512x512 BGR) @ 30-60 Hz                                         |
|  - On-screen Top Speed Slider for direct user throttle control                    |
+----------------------------------------+------------------------------------------+
                                         | TCP Socket (:25001)
                                         v
+-----------------------------------------------------------------------------------+
|                        avis_full_bridge (Bridge Node)                             |
|  - Direct byte-level JPEG parsing (zero string decode overhead)                   |
|  - High-resolution steering actuation (continuous Float32 in [-1.0, 1.0])       |
|  - Discrete servo commands (Int8 in [-6, 6]) for physical hardware compatibility  |
+----------------------------------------+------------------------------------------+
                                         | /camera/image_raw
                                         v
+-----------------------------------------------------------------------------------+
|                   bezier_lane_detector_node (Perception & Math)                   |
|  - Dual Mode:                                                                     |
|      * Mode A: BézierLaneNet Deep Learning Inference (PyTorch CUDA)               |
|      * Mode B: Automatic Perspective CV Centroid Fallback (Sub-millisecond)       |
|  - Cubic Bézier Curve Fitting: B(t) = sum(B_{i,3}(t) * P_i)                       |
|  - Multi-Horizon Lookahead Blending (Near: 0.85, Mid: 0.65, Far: 0.45)            |
|  - Analytical Curvature kappa(t) and Tangent Heading Angle Estimation             |
+----------------------------------------+------------------------------------------+
                                         | /steering_value
                                         v
+-----------------------------------------------------------------------------------+
|                        controller_node (Lateral Actuator)                         |
|  - Instantaneous callback-driven steering output (Zero timer phase delay)         |
|  - Exponential Moving Average (EMA) noise suppression                             |
|  - Continuous Slew-Rate Limiter (prevents abrupt wheel jerks)                     |
|  - Publishes: /steering_cmd (Float32) & /servo (Int8)                             |
+-----------------------------------------------------------------------------------+
```

---

## 📐 Key Innovations & Control Formulation

### 1. Parametric Cubic Bézier Geometry
Each lane boundary and the road centerline are modeled as 2D parametric cubic Bézier curves:
$$B(t) = (1-t)^3 P_0 + 3(1-t)^2 t P_1 + 3(1-t) t^2 P_2 + t^3 P_3, \quad t \in [0, 1]$$

Where:
- $P_0$: Near-hood lane position
- $P_1, P_2$: Internal shape control points
- $P_3$: Far-horizon boundary point

### 2. Analytical Curvature Feedforward
Unlike polynomial approximations that suffer from discretization noise when differentiated, Bézier curves yield exact first and second derivatives at any $t$:
$$B'(t) = 3(1-t)^2 (P_1 - P_0) + 6(1-t)t (P_2 - P_1) + 3t^2 (P_3 - P_2)$$
$$B''(t) = 6(1-t)(P_2 - 2P_1 + P_0) + 6t(P_3 - 2P_2 + P_1)$$

The signed curvature $\kappa(t)$ is calculated analytically:
$$\kappa(t) = \frac{x'(t) y''(t) - y'(t) x''(t)}{\left(x'(t)^2 + y'(t)^2\right)^{3/2}}$$

### 3. Dynamic Perspective Lane Geometry
In forward-facing cameras, apparent road width narrows dramatically with distance. The pipeline dynamically accounts for perspective projection:
$$W(y) = 155.0 + \frac{y - 255.0}{435.0 - 255.0} \times (435.0 - 155.0)$$
This eliminates lane synthesis errors when one boundary is occluded in sharp turns.

### 4. Non-Saturating Multi-Term Lateral Control
The normalized steering command $\delta \in [-1.0, 1.0]$ combines:
$$\delta = K_p \cdot \frac{e_{blended}}{100} + K_d \cdot \frac{de_{filtered}}{dt} + K_i \cdot \int e\,dt + \kappa_{feedforward} + K_{heading} \cdot \theta_{heading}$$
- **Proportional ($K_p = 0.70$)**: Balanced tracking without hunting oscillations.
- **Derivative ($K_d = 1.00$)**: Filtered damping that suppresses lateral chattering.
- **Curvature Feedforward ($K_c = 0.15$)**: Smooth curve entry assistance.
- **Heading Alignment ($K_h = 0.30$)**: Geometric orientation damping.

---

## 📁 Repository Structure

```
Beizer-Algorithm-SelfDriving-car-Simulation/
├── avis_bridge/                       # ROS 2 package for AVIS Engine
│   └── src/avis_bridge/               # Bridge sources & socket protocol
├── avis_bridge_full.py                # Standalone high-throughput TCP bridge node
├── BezierLaneNet/                     # Deep learning backbone & model heads
│   ├── models/                        # ResNet/ConvNeXt Bézier backbone
│   ├── losses/                        # Bézier curve distance loss
│   └── train.py                       # BézierLaneNet training script
├── SelfDriving-jetson Source/         # Jetson & simulation ROS 2 control nodes
│   ├── bezier_lane_detector_node.py   # Primary perception and Bézier extraction node
│   ├── bezier_math.py                 # Pure analytical Bézier mathematical kernel
│   ├── controller_node.py             # Zero-latency lateral actuator node
│   ├── lane_detection_node.py         # Polynomial baseline tracker
│   └── PID-Caculated.py               # Control system transfer function analysis
├── run_sim_pipeline_bezier.py         # One-click launch manager for entire pipeline
├── collect_bezier_dataset.py          # Real-time simulation dataset recorder
├── train_sim_bezier.py                # Pipeline trainer for simulator checkpoints
├── test_bezier_math.py                # Unit test suite for Bézier formulas
├── test_cv_fallback.py                # Unit test suite for computer vision fallback
├── test_node_integration.py           # Integration test suite for ROS 2 graph
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation
```

---

## ⚙️ Prerequisites & Installation

### 1. System Requirements
- **OS**: Ubuntu 24.04 (Noble) or 22.04 (Jammy)
- **ROS 2**: Jazzy Jalisco or Humble Hawksbill
- **Python**: 3.10, 3.11, or 3.12
- **GPU** (Optional): NVIDIA GPU with CUDA for deep learning inference

### 2. Install ROS 2 Vision & OpenCV Dependencies
```bash
sudo apt update
sudo apt install -y ros-$ROS_DISTRO-cv-bridge ros-$ROS_DISTRO-vision-opencv
```

### 3. Clone and Install Python Dependencies
```bash
git clone https://github.com/Mooch3384/Beizer-Algorithm-SelfDriving-car-Simulation.git
cd Beizer-Algorithm-SelfDriving-car-Simulation
pip install -r requirements.txt
```

---

## 🚀 Quick Start

Run the entire autonomous pipeline with a single command:

```bash
python3 run_sim_pipeline_bezier.py
```

### In the AVIS Engine Simulator:
1. Select your desired track (e.g. **Urban Track 1**).
2. Adjust vehicle speed using the **Top Speed** slider in the simulator configuration menu.
3. Click **Start Server** (Port 25001).

The autonomous pipeline will immediately connect, acquire the road lanes, and drive smoothly in the center of the lane.

---

## 🎮 Simulator Configuration

Vehicle speed is **100% user-governed via the AVIS Engine simulator panel**:
- Adjust the `Top Speed` slider at any time during simulation.
- The perception and steering controller will continuously keep the car centered without hunting or leaving the track across all speed ranges.
- To use an existing pre-trained neural network checkpoint:
  ```bash
  python3 run_sim_pipeline_bezier.py --ckpt path/to/model.pth
  ```
  *(If no checkpoint is supplied, the system automatically runs the sub-millisecond Computer Vision Centroid Fallback).*

---

## 🧠 Model Training & Dataset Tools

### Record Simulation Training Data:
```bash
python3 collect_bezier_dataset.py --output_dir ./dataset/sim_run
```

### Train BézierLaneNet on Recorded Data:
```bash
python3 train_sim_bezier.py --data_dir ./dataset/sim_run --epochs 25
```

---

## 🧪 Automated Testing & Verification

Run the built-in mathematical, perception, and integration test suites:

```bash
# 1. Verify analytical Bézier formulas, curvature, and root-finding:
python3 test_bezier_math.py

# 2. Verify computer vision centroid fallback on synthetic and camera frames:
python3 test_cv_fallback.py

# 3. Verify ROS 2 topic communications and inter-node pipeline:
python3 test_node_integration.py
```

---

## 📜 License

This project is licensed under the Apache 2.0 License.
Based on the BézierLaneNet formulation and AVIS Engine ROS 2 Bridge.
