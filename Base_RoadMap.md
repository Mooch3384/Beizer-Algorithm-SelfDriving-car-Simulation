# Base Roadmap - Autonomous Vehicle in Avis Engine Simulator

## Flowchart

```
┌─────────────────────────┐
│   Start Simulation      │
│  (Avis Engine + ROS2)   │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   avis_bridge Node      │
│  Connect to Simulator   │
│  (TCP Socket :25001)    │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Publish Camera Image   │
│  /camera/image_raw      │
│  /camera/camera_info    │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Lane Detection Node    │
│  - Preprocessing        │
│  - Edge/Color Detection │
│  - Lane Line Fitting    │
│  - Crosswalk/Track Det. │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Controller Node        │
│  - Calculate Deviation  │
│  - PID / Stanley Ctrl   │
│  - Speed & Steering Cmd │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Subscribe Topics       │
│  /servo (steering)      │
│  cmd_vel (speed)        │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  avis_bridge Sends      │
│  Commands to Simulator  │
│  (TCP Socket)           │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Loop Back: Receive     │
│  New Frame from Sim     │
└────────────┬────────────┘
             │
             └──────────► (Repeat)
```

## Project Phases

### Phase 1: Infrastructure Setup
- [ ] Verify Avis Engine simulator runs correctly
- [ ] Build and test avis_bridge node
- [ ] Confirm camera image topic publishing
- [ ] Confirm command subscription working

### Phase 2: Perception (Lane & Track Detection)
- [ ] Implement image preprocessing (blur, threshold, ROI)
- [ ] Develop lane line detection algorithm
- [ ] Add crosswalk / track boundary detection
- [ ] Test perception node independently with recorded bags

### Phase 3: Control
- [ ] Implement lateral controller (PID or Stanley)
- [ ] Implement longitudinal controller (speed regulation)
- [ ] Tune control parameters in simulation
- [ ] Handle edge cases (lost lane, sharp turns)

### Phase 4: Integration & Testing
- [ ] Connect perception → controller → bridge (closed loop)
- [ ] End-to-end testing on simple track
- [ ] Performance benchmarking (FPS, latency, deviation)
- [ ] Parameter tuning and optimization

### Phase 5: Advanced Features (Optional)
- [ ] Traffic sign recognition (YOLO / AprilTag)
- [ ] State machine for decision making
- [ ] SLAM / VSLAM integration
- [ ] Ackermann control standardization
- [ ] Transfer to Jetson Orin Nano hardware

## Key ROS2 Topics

| Topic | Type | Direction | Description |
|-------|------|-----------|-------------|
| `/camera/image_raw` | sensor_msgs/Image | Bridge → Perception | Raw camera frame |
| `/camera/camera_info` | sensor_msgs/CameraInfo | Bridge → Perception | Camera intrinsics |
| `/servo` | std_msgs/Int8 | Controller → Bridge | Steering command [-6,6] |
| `cmd_vel` | std_msgs/Int8 | Controller → Bridge | Speed command [-6,6] |
| `/cmd_vel_twist` | geometry_msgs/Twist | Controller → Bridge | Alternative velocity cmd |

## Architecture Nodes

1. **avis_bridge** - TCP communication with Avis Engine simulator
2. **lane_detection_node** - Image processing and lane/track detection
3. **controller_node** - Path following and speed/steering computation
4. **image_filter_node** - Optional preprocessing/filtering stage

</parameter>