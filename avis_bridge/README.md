# ROS 2 Avis Engine Simulator Bridge (`avis_bridge`)

[![ROS 2 Version](https://img.shields.io/badge/ROS2-Jazzy%20Jalisco-orange.svg)](https://docs.ros.org/en/jazzy/)
[![Python Version](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A high-performance, resilient **ROS 2 Jazzy** bridge node connecting the **FIRA Cup Avis Engine Simulator** with the ROS 2 autonomous vehicle software stack.

---

## 1. Project Overview

### What This Project Does
This package provides a bidirectional communications bridge between the **Avis Engine Simulator** (a 3D autonomous vehicle simulator used in FIRA Cup competitions) and **ROS 2**. It continuously streams simulated camera frames into the ROS 2 ecosystem while consuming motor and steering commands from autonomous driving nodes and forwarding them back to the simulator in real time.

### Why This Bridge Exists
Developing autonomous vehicles requires a safe, reproducible simulation environment before deploying algorithms on physical hardware (such as a Jetson Orin Nano). This bridge abstracts the low-level TCP socket protocol of Avis Engine, exposing clean ROS 2 topics and parameters so developers can build and evaluate:
- Lane detection & road boundary tracking
- Traffic sign & object detection
- Decision-making state machines
- Control & path-following algorithms

---

## 2. Features

- **ROS 2 Jazzy & Python 3.12 Support**: Built specifically to conform to standard ROS 2 Jazzy conventions and modern Python 3.12 practices.
- **Resilient TCP Socket Management**: Background worker loop automatically handles connection establishment, loss detection, and seamless auto-reconnection without crashing the ROS 2 spin loop.
- **Synchronized Camera Streaming**: Converts incoming JPEG/PNG Base64 socket payloads to standard `sensor_msgs/msg/Image` (`bgr8` encoding).
- **Camera Calibration Support**: Publishes synchronized `sensor_msgs/msg/CameraInfo` messages for downstream 3D computer vision and Inverse Perspective Mapping (IPM).
- **Thread-Safe Command Pipeline**: Uses internal `threading.Lock()` mutexes to guarantee thread-safe data sharing between ROS subscriber callbacks and the TCP socket thread.
- **Flexible Command Interfaces**: Accepts legacy integer control commands (`std_msgs/msg/Int8`), floating-point commands (`std_msgs/msg/Float32`), and standard ROS 2 velocity commands (`geometry_msgs/msg/Twist`).
- **Dynamic ROS 2 Parameters**: Fully configurable IP, port, topic names, scaling factors, and frame IDs, supporting runtime updates via `ros2 param set` and `rqt_reconfigure`.

---

## 3. System Architecture

```
 ┌──────────────────────┐         TCP Socket (Port 25001)        ┌──────────────────────────────┐
 │                      │  <------- Control Commands --------  │   avis_bridge (bridge_node)  │
 │  Avis Engine         │     (Speed, Steering, Angle)         │                              │
 │  Simulator           │                                      │  ┌────────────────────────┐  │
 │                      │  -------- Sensor & Image --------->  │  │ _communication_loop   │  │
 └──────────────────────┘         (XML + Base64)               │  └───────────┬────────────┘  │
                                                                               │              │
                                                                    ROS 2 Topics              │
                                                                               │              │
 ┌──────────────────────────┐             ┌────────────────────────────────────┼──────────────┘
 │ Lane & Sign Detection    │ <--- Image -│  /camera/image_raw (sensor_msgs/Image)│
 └────────────┬─────────────┘             │  /camera/camera_info (sensor_msgs/CameraInfo)
              │                           └───────────────────────────────────────────────────
              ▼
 ┌──────────────────────────┐
 │ Decision State Machine   │
 └────────────┬─────────────┘
              │                           ┌───────────────────────────────────────────────────
              └────── Servo / Speed ----> │  /servo & cmd_vel (std_msgs/Int8)                 │
                                          │  /cmd_vel_twist (geometry_msgs/Twist)             │
                                          └───────────────────────────────────────────────────
```

---

## 4. Project Structure

```text
ros2_avis/
├── src/
│   └── avis_bridge/
│       ├── avis_bridge/
│       │   ├── __init__.py
│       │   ├── avisengine.py    # Low-level TCP socket communication layer (Car class)
│       │   ├── bridge_node.py   # Main ROS 2 Bridge Node (AvisBridgeNode)
│       │   └── utils.py         # Helper utilities
│       ├── package.xml          # ROS 2 package manifest
│       ├── setup.py             # Python package installer & entry points
│       └── setup.cfg            # Package build configuration
└── README.md                    # Project documentation
```

### Important Files
- **`bridge_node.py`**: Main ROS 2 node handling publishers, subscribers, parameters, thread safety, and data conversion.
- **`avisengine.py`**: Communication interface defining the `Car` class, managing raw TCP socket packets, string encoding, and XML response parsing.
- **`utils.py`**: Supplementary helper functions.

---

## 5. Installation

### Prerequisites
- **OS**: Linux (Ubuntu 24.04 Noble Numbat recommended)
- **ROS Version**: ROS 2 Jazzy Jalisco
- **Python**: 3.12+

### System Dependencies
Ensure ROS 2 dependencies and OpenCV are installed:
```bash
sudo apt update
sudo apt install -y ros-jazzy-rclpy ros-jazzy-sensor-msgs ros-jazzy-std-msgs ros-jazzy-geometry-msgs ros-jazzy-cv-bridge python3-opencv
```

> [!IMPORTANT]
> **NumPy Compatibility Note**: ROS 2 Jazzy's C++ bindings (`cv_bridge`) require `numpy<2` (specifically NumPy `1.26.4`). If NumPy 2.x is installed in your local Python environment, downgrade it using:
> ```bash
> python3 -m pip install "numpy<2" --break-system-packages
> ```

---

## 6. Running the Project

### Method 1: Direct Execution (Python Script)
You can start the bridge node directly using Python without building the workspace:
```bash
cd ~/Desktop/My_WorkSpacae/Simulator/ros2_avis/src/avis_bridge/avis_bridge
python3 bridge_node.py
```

### Method 2: ROS 2 Package Execution (`ros2 run`)
Build and run using standard ROS 2 `colcon` workflow:
```bash
cd ~/Desktop/My_WorkSpacae/Simulator/ros2_avis
colcon build --packages-select avis_bridge
source install/setup.bash
ros2 run avis_bridge bridge_node
```

---

## 7. ROS 2 Interface Documentation

### Published Topics

| Topic Name | Message Type | QoS Profile | Description |
| :--- | :--- | :--- | :--- |
| `camera/image_raw` | `sensor_msgs/msg/Image` | Best Effort, Volatile, Depth 1 | Live simulated camera frame (`bgr8` encoding). |
| `camera/camera_info` | `sensor_msgs/msg/CameraInfo` | Best Effort, Volatile, Depth 1 | Synced camera intrinsic matrix ($K$) and projection matrix ($P$). |

### Subscribed Topics

| Topic Name | Message Type | QoS Profile | Description |
| :--- | :--- | :--- | :--- |
| `/servo` | `std_msgs/msg/Int8` | Reliable, Volatile, Depth 5 | Steering command input (Default Range: `[-6, 6]`). |
| `cmd_vel` | `std_msgs/msg/Int8` | Reliable, Volatile, Depth 5 | Speed command input (Default Range: `[-6, 6]`). |
| `/cmd_vel_twist` | `geometry_msgs/msg/Twist` | Reliable, Volatile, Depth 5 | Standard ROS 2 velocity input (`linear.x` = speed, `angular.z` = steering). |

---

## 8. Configuration & Parameters

The node exposes several parameters that can be overridden at startup or reconfigured at runtime:

### Available Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `simulator_ip` | `string` | `'127.0.0.1'` | Avis Engine server IP address. |
| `simulator_port` | `integer` | `25001` | Avis Engine TCP port. |
| `image_topic` | `string` | `'camera/image_raw'` | Topic for camera raw image publishing. |
| `camera_info_topic` | `string` | `'camera/camera_info'` | Topic for camera info calibration data. |
| `servo_topic` | `string` | `'/servo'` | Topic for discrete steering input. |
| `cmd_vel_topic` | `string` | `'cmd_vel'` | Topic for discrete speed input. |
| `camera_frame_id` | `string` | `'avis_camera'` | TF Frame ID associated with image headers. |
| `max_steering_avis` | `integer` | `100` | Max steering value accepted by Avis Engine (`[-100, 100]`). |
| `max_speed_avis` | `integer` | `350` | Max speed value accepted by Avis Engine (`[-350, 350]`). |
| `max_servo_input` | `integer` | `6` | Max discrete value for `/servo` topic (`[-6, 6]`). |
| `max_cmd_vel_input` | `integer` | `6` | Max discrete value for `cmd_vel` topic (`[-6, 6]`). |
| `sensor_angle` | `integer` | `30` | Lidar/Sensor ray spread angle in simulator (degrees). |
| `publish_camera_info` | `boolean` | `true` | Enable/disable publishing `CameraInfo` messages. |
| `reconnect_interval_sec` | `float` | `2.0` | Delay between TCP reconnect attempts (seconds). |

### Overriding Parameters at Launch
```bash
ros2 run avis_bridge bridge_node --ros-args -p simulator_ip:="127.0.0.1" -p max_speed_avis:=300
```

### Dynamic Runtime Reconfiguration
```bash
ros2 param set /avis_bridge_node max_speed_avis 250
```

---

## 9. How It Works Internally

### `bridge_node.py` Architecture
1. **Parameter Initialization**: Declares all ROS 2 parameters and registers `_on_parameter_change` to support live dynamic re-configuration.
2. **Subscription Mutex**: Callbacks (`_servo_callback`, `_cmd_vel_callback`, `_twist_callback`) update internal state variables (`_current_servo`, `_current_cmd_vel`) protected by a `threading.Lock()`.
3. **Background Worker Loop (`_communication_loop`)**:
   - Checks `car.is_connected`. If disconnected, attempts `car.connect()` every `reconnect_interval_sec`.
   - Safely reads raw commands, normalizes them to `[-1.0, 1.0]`, and scales them to Avis Engine limits.
   - Calls `car.getData()`, triggering the socket update cycle.
   - Decodes the raw image and publishes synced `Image` and `CameraInfo` ROS 2 messages.

### `avisengine.py` Socket Communication
1. **Connection**: Establishes a standard TCP socket via Python's `socket.socket()`.
2. **Payload Formatting**: Formats command payload:
   `Speed:{speed},Steering:{steering},ImageStatus:{1},SensorStatus:{1},GetSpeed:{1},SensorAngle:{30}`
3. **Byte Parsing**: `recvall()` continuously appends byte chunks until finding `b"<EOF>"`.
4. **Data Extraction**: Extracts tags using regular expressions:
   - `<image>...base64...</image>` $\rightarrow$ Base64 decoded $\rightarrow$ `cv2.imdecode()` $\rightarrow$ OpenCV BGR Array
   - `<sensor>v1,v2,v3</sensor>` $\rightarrow$ Parsed integer ray distance array
   - `<speed>v</speed>` $\rightarrow$ Integer vehicle speed feedback

---

## 10. Future Development & Roadmap

The current bridge is designed to serve as the foundation for a full autonomous vehicle stack:

1. **Lane Detection Node**: Consume `camera/image_raw`, compute lane centerlines, and output steering corrections to `/servo`.
2. **Traffic Sign Recognition Node**: Run YOLOv8 / AprilTag detection on `camera/image_raw` to detect stop signs and traffic lights.
3. **State Machine Node**: Integrate high-level decision logic (lane following, crosswalk stop, speed control) publishing to `cmd_vel` and `/servo`.
4. **Ackermann Control Standard**: Transition control commands to `ackermann_msgs/msg/AckermannDriveStamped` (steering angle in radians, speed in m/s) to streamline physical deployment on real RC vehicles (e.g., Jetson Orin Nano).

---

## 11. Development Notes For Future AI Code Review

> [!NOTE]
> **Instructions for AI Assistants & Automated Code Reviews**:

1. **Core Architectural Invariants**:
   - **Do NOT remove thread locking** (`self._lock`) around `_current_servo` and `_current_cmd_vel`. ROS 2 callbacks run on executor threads while socket communication runs on a separate worker thread.
   - **Do NOT remove socket auto-reconnection** in `_communication_loop`. The node must gracefully wait when Avis Engine restarts.
   - **Do NOT replace native binary search (`b"<EOF>" in data`)** in `avisengine.py` with custom iterative string search loops; native byte matching is essential for frame rate performance.

2. **Dependency Constraints**:
   - Maintain compatibility with **ROS 2 Jazzy** and **Python 3.12**.
   - Keep `numpy<2` compatibility in mind when importing `cv_bridge`.

3. **Code Style & Structure**:
   - Follow PEP 8 guidelines.
   - Preserve parameter declaration patterns and dynamic parameter callbacks (`add_on_set_parameters_callback`).
