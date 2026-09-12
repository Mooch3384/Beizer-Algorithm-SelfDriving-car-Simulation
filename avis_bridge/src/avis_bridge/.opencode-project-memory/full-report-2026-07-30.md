# avis_bridge — Full Architecture Report

**Date:** 2026-07-30  
**Status:** All files read fresh (no staleness).  
**Git repo:** No (parent is not a git repo).

---

## Package Metadata (`package.xml`)

| Field | Value |
|-------|-------|
| Name | `avis_bridge` |
| Version | `0.0.0` |
| Format | 3 (ament_python) |
| Maintainer | david <Davidrashidi0001311pc@gmail.com> |
| Description | `TODO: Package description` (placeholder) |
| License | `TODO: License declaration` (placeholder) |

**Dependencies:**
- `rclpy` (build + exec, via `<depend>`)
- `sensor_msgs` (exec only, via `<exec_depend>`)
- `std_msgs` (exec only, via `<exec_depend>`)
- `cv_bridge` (exec only, via `<exec_depend>`)

**Test dependencies:**
- `ament_copyright`, `ament_flake8`, `ament_pep257`, `python3-pytest`

---

## Entry Points (`setup.py`)

Single console script:
```
bridge_node = avis_bridge.bridge_node:main
```
Invoked as `bridge_node` after `colcon build`.

---

## Source Modules

### `avis_bridge/__init__.py`
Empty file — package marker only.

### `avis_bridge/bridge_node.py` — Main ROS 2 node (187 lines)
**Class:** `AvisBridgeNode(Node)`

**Node name:** `avis_bridge_node`

**Publishers:**
| Topic | Type | QoS |
|-------|------|-----|
| `camera/image_raw` | `sensor_msgs/Image` | depth=10 (default) |

**Subscribers:**
| Topic | Type | QoS |
|-------|------|-----|
| `/servo` | `std_msgs/Int8` | BEST_EFFORT, VOLATILE, KEEP_LAST(depth=1) |

**Data flow:**
```
AVIS Simulator <--socket--> bridge_node <--ROS 2--> pipeline
     camera ───────────────> camera/image_raw ─────> image_filter_node
     steering <─────────────── /servo <────────────── state_node
```

**Key constants:**
- `SIMULATOR_IP = "127.0.0.1"`
- `SIMULATOR_PORT = 25001`
- `DEFAULT_SPEED = 0` (constant forward speed)
- `SENSOR_ANGLE = 30` degrees
- `STARTUP_DELAY = 3` seconds

**Threading:** A background daemon thread (`_data_loop`) continuously pulls frames from the simulator via `Car.getData()`, maps `/servo` Int8 [-6,6] → steering [-100,100], and publishes `sensor_msgs/Image` (bgr8 encoding) on `camera/image_raw`. First 4 frames are skipped (warmup).

**Cleanup:** `destroy_node()` sets `_running=False`, calls `car.stop()`, then `super().destroy_node()`.

### `avis_bridge/avisengine.py` — AVIS Engine socket client (141 lines)
**Class:** `Car`  
**Author:** Amirmohammad Zarif (AVIS Engine)  
**Copyright:** 2025 AVIS Engine

**Protocol:** Plain TCP socket. Sends comma-separated key:value string:
```
Speed:{},Steering:{},ImageStatus:{},SensorStatus:{},GetSpeed:{},SensorAngle:{}
```
Receives a response delimited by `<EOF>`, parsed for tags:
- `<image>...</image>` → base64-decoded JPEG → OpenCV BGR image
- `<sensor>...</sensor>` → distance array (defaults to `[1500,1500,1500]` on failure)
- `<speed>...</speed>` → current speed int

**Key methods:**
- `connect(server, port)` — TCP connect with 5s timeout
- `getData()` — sends current state, receives image+sensors+speed
- `setSteering(v)`, `setSpeed(v)`, `setSensorAngle(a)` — send-only
- `getImage()` → numpy array (OpenCV BGR)
- `getSensors()` → list[int]
- `getSpeed()` → int
- `stop()` — set speed+steering to 0, send "stop", close socket

**Note:** Class-level attributes (`steering_value`, `speed_value`, etc.) act as defaults — these are class variables, not instance variables, which means multiple `Car` instances share the same initial values (mild quirk, but bridge_node uses only one instance).

### `avis_bridge/utils.py` — AVIS Engine utilities (73 lines)
**Copyright:** 2025 AVIS Engine

**Functions:**
- `stringToImage(base64_string)` — PIL image from base64 (unused in bridge_node)
- `BGRtoRGB(image)` — PIL→numpy RGB conversion (unused in bridge_node)
- `KMPSearch(pat, txt)` — KMP substring search (used by `avisengine.Car.recvall` to find `<EOF>` delimiter)
- `computeLPS(pat, M, lps)` — LPS table for KMP

---

## Tests (3 files, all ROS 2 boilerplate linters)

All in `test/`:
1. `test_flake8.py` — Flake8 linting via `ament_flake8` (active)
2. `test_copyright.py` — Copyright header check via `ament_copyright` (**SKIPPED** — `@pytest.mark.skip` with reason "No copyright header has been placed")
3. `test_pep257.py` — PEP 257 docstring check via `ament_pep257` (active)

No functional/unit tests exist for the bridge logic.

---

## Build & Test Commands

**Build:**
```bash
cd /home/david/Desktop/My_WorkSpacae/Simulator/ros2_avis
colcon build --packages-select avis_bridge
source install/setup.bash
```

**Run:**
```bash
ros2 run avis_bridge bridge_node
```

**Test:**
```bash
colcon test --packages-select avis_bridge
# or individually:
pytest test/test_flake8.py
pytest test/test_pep257.py
```

**Note:** There is no `CMakeLists.txt` — this is a pure `ament_python` package. No `verify.sh` or custom acceptance test.

---

## Quirks & Observations

1. **Placeholder metadata:** Both `description` and `license` in `package.xml` are TODO stubs.
2. **Empty resource file:** `resource/avis_bridge` is an empty file (required by ament index).
3. **Class-level defaults in `Car`:** `steering_value`, `speed_value`, `sensor_angle` are class variables, not instance variables — shared across instances until reassigned. Only one `Car` instance is used, so this is benign.
4. **No error recovery on socket failure:** If the simulator disconnects mid-loop, `_data_loop` catches the exception and sleeps 0.5s, but `self._connected` stays True and `car` is not reconnected.
5. **QoS mismatch:** `camera/image_raw` publisher uses default QoS (depth=10), while `/servo` subscriber uses BEST_EFFORT/KEEP_LAST(1). This is intentional per the comment ("matches the rest of the pipeline").
6. **Copyright skipped:** The `test_copyright.py` test is explicitly skipped — AVIS Engine files (`avisengine.py`, `utils.py`) have their own copyright headers (2025 AVIS Engine), but the ROS-generated files don't have Apache 2.0 headers.
7. **No launch file:** No `.launch.py` or `.yaml` launch configuration exists.
8. **sys.path hack:** `bridge_node.py` inserts its own directory into `sys.path` (line 31) to import `avisengine` and `utils` as top-level modules — this works when run directly but could cause import confusion if another package has modules with the same names.