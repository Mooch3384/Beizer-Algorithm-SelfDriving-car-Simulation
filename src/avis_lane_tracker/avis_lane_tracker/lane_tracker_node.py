import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, TransformStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge, CvBridgeError
from tf2_ros import TransformBroadcaster

import numpy as np
import cv2
import math
from typing import List, Tuple

from avis_lane_tracker.ipm import IPMTransformer
from avis_lane_tracker.lane_detector import EgoLaneDetector


class LaneTrackerNode(Node):
    """
    Main ROS 2 Node for Real-Time Online Lane Tracking, IPM Perspective Transformation,
    Trajectory Plotting, Local Mapping, and Visual Debugging.
    """

    def __init__(self):
        super().__init__('lane_tracker_node')

        # ----------------------------------------------------
        # Declare ROS 2 Parameters
        # ----------------------------------------------------
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('trajectory_topic', '/robot_trajectory')
        self.declare_parameter('markers_topic', '/lane/markers')
        self.declare_parameter('bev_debug_topic', '/lane/bev_debug')

        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')

        # IPM Calibration Matrix Points (Camera -> BEV)
        self.declare_parameter('bev_width', 640)
        self.declare_parameter('bev_height', 480)

        # Sliding Window & Metric Scale Parameters
        self.declare_parameter('nwindows', 10)
        self.declare_parameter('margin', 40)
        self.declare_parameter('minpix', 50)
        self.declare_parameter('meters_per_pixel_x', 3.7 / 200.0)
        self.declare_parameter('meters_per_pixel_y', 10.0 / 480.0)

        # Read Parameters
        self.camera_topic = self.get_parameter('camera_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.trajectory_topic = self.get_parameter('trajectory_topic').value
        self.markers_topic = self.get_parameter('markers_topic').value
        self.bev_debug_topic = self.get_parameter('bev_debug_topic').value

        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value

        bev_w = self.get_parameter('bev_width').value
        bev_h = self.get_parameter('bev_height').value

        nwindows = self.get_parameter('nwindows').value
        margin = self.get_parameter('margin').value
        minpix = self.get_parameter('minpix').value
        m_per_px_x = self.get_parameter('meters_per_pixel_x').value
        m_per_px_y = self.get_parameter('meters_per_pixel_y').value

        # ----------------------------------------------------
        # Initialize Core Vision & Processing Modules
        # ----------------------------------------------------
        src_pts = [
            [180, 330],  # Top-Left
            [460, 330],  # Top-Right
            [610, 470],  # Bottom-Right
            [30, 470]    # Bottom-Left
        ]
        dst_pts = [
            [100, 0],
            [bev_w - 100, 0],
            [bev_w - 100, bev_h],
            [100, bev_h]
        ]

        self.ipm = IPMTransformer(src_points=src_pts, dst_points=dst_pts, out_shape=(bev_w, bev_h))
        self.detector = EgoLaneDetector(
            nwindows=nwindows,
            margin=margin,
            minpix=minpix,
            meters_per_pixel_x=m_per_px_x,
            meters_per_pixel_y=m_per_px_y
        )
        self.bridge = CvBridge()

        # Data structures for driven trajectory path
        self.path_msg = Path()
        self.path_msg.header.frame_id = self.odom_frame

        # ----------------------------------------------------
        # ROS 2 Subscribers & Publishers
        # ----------------------------------------------------
        self.image_sub = self.create_subscription(
            Image,
            self.camera_topic,
            self._image_callback,
            10
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self._odom_callback,
            10
        )

        self.path_pub = self.create_publisher(Path, self.trajectory_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, self.markers_topic, 10)
        self.bev_debug_pub = self.create_publisher(Image, self.bev_debug_topic, 10)

        # Dynamic TF Broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info("LaneTrackerNode initialized successfully.")
        self.get_logger().info(f"Subscribed to: {self.camera_topic}, {self.odom_topic}")
        self.get_logger().info(f"Publishing to: {self.trajectory_topic}, {self.markers_topic}, {self.bev_debug_topic}")

    def _odom_callback(self, msg: Odometry):
        """
        Odometry callback: Update vehicle driven path trajectory and broadcast TF (odom -> base_link).
        """
        now = self.get_clock().now().to_msg()

        # 1. Update Path trajectory
        pose_stamped = PoseStamped()
        pose_stamped.header = msg.header
        pose_stamped.header.frame_id = self.odom_frame
        pose_stamped.pose = msg.pose.pose

        self.path_msg.header.stamp = now
        self.path_msg.poses.append(pose_stamped)
        self.path_pub.publish(self.path_msg)

        # 2. Broadcast TF (odom -> base_link)
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame

        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z

        t.transform.rotation = msg.pose.pose.orientation

        self.tf_broadcaster.sendTransform(t)

    def _image_callback(self, msg: Image):
        """
        Monocular Front Camera Callback: Performs real-time IPM top-down warping,
        ego-lane segmentation, sliding window polynomial tracking, local marker publishing,
        and BEV visual debug feed streaming.
        """
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge error converting image: {e}")
            return

        if cv_img is None or cv_img.size == 0:
            return

        # 1. Inverse Perspective Mapping (IPM) Homography Transformation
        try:
            bev_img = self.ipm.transform(cv_img)
        except Exception as e:
            self.get_logger().error(f"IPM Transformation failed: {e}")
            return

        # 2. Ego-Lane Segmentation & Adaptive Binarization
        binary_bev = self.detector.binarize(bev_img)

        # 3. Sliding Window Histogram & Polynomial Curve Fitting
        detection_results = self.detector.detect_lane(binary_bev)

        # 4. Publish Processed BEV Debug Frame
        debug_img = detection_results["debug_img"]
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
            debug_msg.header = msg.header
            self.bev_debug_pub.publish(debug_msg)
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge debug frame error: {e}")

        # 5. Local Mapping: Convert BEV Polynomial Curves to Base Link Frame Points
        img_h, img_w = binary_bev.shape
        ploty = detection_results["ploty"]
        left_fitx = detection_results["left_fitx"]
        right_fitx = detection_results["right_fitx"]
        center_fitx = detection_results["center_fitx"]

        left_points_base = self.detector.bev_to_base_link(left_fitx, ploty, (img_h, img_w))
        right_points_base = self.detector.bev_to_base_link(right_fitx, ploty, (img_h, img_w))
        center_points_base = self.detector.bev_to_base_link(center_fitx, ploty, (img_h, img_w))

        # 6. Publish RViz 2 Markers for Ego-Lane Boundaries & Centerline Waypoints
        self._publish_rviz_markers(msg.header.stamp, left_points_base, right_points_base, center_points_base)

    def _publish_rviz_markers(
        self,
        stamp,
        left_pts: List[Tuple[float, float]],
        right_pts: List[Tuple[float, float]],
        center_pts: List[Tuple[float, float]]
    ):
        """
        Construct and publish RViz MarkerArray for local lane boundaries and centerline.
        """
        marker_array = MarkerArray()

        # Marker 1: Left Lane Line (Yellow LINE_STRIP)
        left_marker = Marker()
        left_marker.header.stamp = stamp
        left_marker.header.frame_id = self.base_frame
        left_marker.ns = "left_lane"
        left_marker.id = 0
        left_marker.type = Marker.LINE_STRIP
        left_marker.action = Marker.ADD
        left_marker.scale.x = 0.15  # Line width
        left_marker.color.r = 1.0
        left_marker.color.g = 1.0
        left_marker.color.b = 0.0
        left_marker.color.a = 1.0

        for x_m, y_m in left_pts:
            p = Point()
            p.x = x_m
            p.y = y_m
            p.z = 0.0
            left_marker.points.append(p)

        marker_array.markers.append(left_marker)

        # Marker 2: Right Lane Line (White/Cyan LINE_STRIP)
        right_marker = Marker()
        right_marker.header.stamp = stamp
        right_marker.header.frame_id = self.base_frame
        right_marker.ns = "right_lane"
        right_marker.id = 1
        right_marker.type = Marker.LINE_STRIP
        right_marker.action = Marker.ADD
        right_marker.scale.x = 0.15
        right_marker.color.r = 0.0
        right_marker.color.g = 1.0
        right_marker.color.b = 1.0
        right_marker.color.a = 1.0

        for x_m, y_m in right_pts:
            p = Point()
            p.x = x_m
            p.y = y_m
            p.z = 0.0
            right_marker.points.append(p)

        marker_array.markers.append(right_marker)

        # Marker 3: Ego Centerline (Green LINE_STRIP)
        center_marker = Marker()
        center_marker.header.stamp = stamp
        center_marker.header.frame_id = self.base_frame
        center_marker.ns = "centerline"
        center_marker.id = 2
        center_marker.type = Marker.LINE_STRIP
        center_marker.action = Marker.ADD
        center_marker.scale.x = 0.2
        center_marker.color.r = 0.0
        center_marker.color.g = 1.0
        center_marker.color.b = 0.0
        center_marker.color.a = 1.0

        for x_m, y_m in center_pts:
            p = Point()
            p.x = x_m
            p.y = y_m
            p.z = 0.0
            center_marker.points.append(p)

        marker_array.markers.append(center_marker)

        # Marker 4: Waypoint Spheres along Centerline
        spheres_marker = Marker()
        spheres_marker.header.stamp = stamp
        spheres_marker.header.frame_id = self.base_frame
        spheres_marker.ns = "centerline_waypoints"
        spheres_marker.id = 3
        spheres_marker.type = Marker.SPHERE_LIST
        spheres_marker.action = Marker.ADD
        spheres_marker.scale.x = 0.3
        spheres_marker.scale.y = 0.3
        spheres_marker.scale.z = 0.3
        spheres_marker.color.r = 1.0
        spheres_marker.color.g = 0.0
        spheres_marker.color.b = 0.0
        spheres_marker.color.a = 0.9

        # Sample waypoints every 5th point
        for i in range(0, len(center_pts), 10):
            x_m, y_m = center_pts[i]
            p = Point()
            p.x = x_m
            p.y = y_m
            p.z = 0.1
            spheres_marker.points.append(p)

        marker_array.markers.append(spheres_marker)

        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = LaneTrackerNode()
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
