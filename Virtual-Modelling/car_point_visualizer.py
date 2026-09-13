#!/usr/bin/env python3
"""
car_point_visualizer.py
ROS 2 Visualizer Node for Point-Robot Model & Metric Trajectory in RViz2.

Visualizes:
1. Vehicle Point-Mass Model: Red sphere at (0, 0, 0) in base_link frame (/car_point_marker).
2. Trajectory History: Path traveled in odom frame (/car_trajectory).
3. Planned 50m Trajectory: Cyan 3D Line-Strip marker in base_link frame (/planned_trajectory_marker).
4. Lookahead Target Point: Bright green preview sphere marker in base_link frame (/target_point_marker).
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Point, TransformStamped
from visualization_msgs.msg import Marker
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


class CarPointVisualizer(Node):
    def __init__(self):
        super().__init__('car_point_visualizer')

        self.declare_parameter('odom_topic', '/odom')
        odom_topic = self.get_parameter('odom_topic').value

        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ── Subscribers ──────────────────────────────────────────────────────
        # 1. Odometry from vehicle simulator / bridge
        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, odom_qos
        )

        # 2. Planned 50m ground metric trajectory from Bézier Lane Detector
        self.planned_sub = self.create_subscription(
            Path, '/planned_trajectory', self.planned_trajectory_callback, sensor_qos
        )

        # ── Publishers for RViz2 ─────────────────────────────────────────────
        self.marker_pub = self.create_publisher(Marker, '/car_point_marker', 10)
        self.path_pub = self.create_publisher(Path, '/car_trajectory', 10)
        self.plan_marker_pub = self.create_publisher(Marker, '/planned_trajectory_marker', 10)
        self.target_marker_pub = self.create_publisher(Marker, '/target_point_marker', 10)

        # Trajectory history in odom frame
        self.path_msg = Path()
        self.path_msg.header.frame_id = 'odom'

        # Broadcast static transform map -> odom so Fixed Frame [map] in RViz2 works out-of-the-box
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        self._publish_static_map_tf()

        self.get_logger().info(f"Car Point Visualizer listening to {odom_topic} and /planned_trajectory ...")

    def _publish_static_map_tf(self):
        """Broadcasts static identity transform map -> odom."""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.static_tf_broadcaster.sendTransform(t)

    def odom_callback(self, msg: Odometry):
        # 1. Save and publish historical path traveled
        pose_stamped = PoseStamped()
        pose_stamped.header = msg.header
        pose_stamped.pose = msg.pose.pose
        self.path_msg.header.stamp = msg.header.stamp
        self.path_msg.poses.append(pose_stamped)

        if len(self.path_msg.poses) > 1000:
            self.path_msg.poses.pop(0)

        self.path_pub.publish(self.path_msg)

        # 2. Point-Robot Vehicle Marker (Red sphere at base_link center)
        car_marker = Marker()
        car_marker.header.stamp = msg.header.stamp
        car_marker.header.frame_id = 'base_link'
        car_marker.ns = 'vehicle_model'
        car_marker.id = 0
        car_marker.type = Marker.SPHERE
        car_marker.action = Marker.ADD

        car_marker.pose.position.x = 0.0
        car_marker.pose.position.y = 0.0
        car_marker.pose.position.z = 0.0
        car_marker.pose.orientation.w = 1.0

        car_marker.scale.x = 0.35
        car_marker.scale.y = 0.35
        car_marker.scale.z = 0.35

        car_marker.color.r = 1.0
        car_marker.color.g = 0.05
        car_marker.color.b = 0.05
        car_marker.color.a = 1.0

        self.marker_pub.publish(car_marker)

    def planned_trajectory_callback(self, msg: Path):
        if not msg.poses:
            return

        # 1. Planned Trajectory 3D Line Strip (Cyan #00FFFF)
        line_marker = Marker()
        line_marker.header = msg.header
        line_marker.ns = 'planned_trajectory'
        line_marker.id = 1
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD

        line_marker.scale.x = 0.15  # Line thickness 15 cm
        line_marker.color.r = 0.0
        line_marker.color.g = 1.0
        line_marker.color.b = 1.0
        line_marker.color.a = 0.95

        for pose in msg.poses:
            p = Point()
            p.x = pose.pose.position.x
            p.y = pose.pose.position.y
            p.z = pose.pose.position.z
            line_marker.points.append(p)

        self.plan_marker_pub.publish(line_marker)

        # 2. Lookahead Target Marker (Bright Green Sphere at ~12m horizon)
        target_marker = Marker()
        target_marker.header = msg.header
        target_marker.ns = 'target_point'
        target_marker.id = 2
        target_marker.type = Marker.SPHERE
        target_marker.action = Marker.ADD

        target_marker.scale.x = 0.40
        target_marker.scale.y = 0.40
        target_marker.scale.z = 0.40
        target_marker.color.r = 0.0
        target_marker.color.g = 1.0
        target_marker.color.b = 0.0
        target_marker.color.a = 1.0

        # Pick target point around X = 12m
        chosen_pose = msg.poses[min(len(msg.poses) - 1, len(msg.poses) // 3)]
        for pose in msg.poses:
            if pose.pose.position.x >= 12.0:
                chosen_pose = pose
                break

        target_marker.pose = chosen_pose.pose
        self.target_marker_pub.publish(target_marker)


def main(args=None):
    rclpy.init(args=args)
    node = CarPointVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()