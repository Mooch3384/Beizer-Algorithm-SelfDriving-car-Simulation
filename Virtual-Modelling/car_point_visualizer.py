#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker

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

        # سابسکرایب به دیتای حرکتی خودرو
        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, odom_qos
        )

        # پابلیشرهای مخصوص RViz2
        self.marker_pub = self.create_publisher(Marker, '/car_point_marker', 10)
        self.path_pub = self.create_publisher(Path, '/car_trajectory', 10)

        # ساختار نگهداری خط مسیر (Trajectory)
        self.path_msg = Path()
        self.path_msg.header.frame_id = 'odom'

        self.get_logger().info(f"Car Point Visualizer listening to {odom_topic} ...")

    def odom_callback(self, msg: Odometry):
        # ۱. ذخیره و انتشار مسیر طی شده (Trajectory)
        pose_stamped = PoseStamped()
        pose_stamped.header = msg.header
        pose_stamped.pose = msg.pose.pose
        self.path_msg.header.stamp = msg.header.stamp
        self.path_msg.poses.append(pose_stamped)
        
        # محدود کردن طول تاریخچه به ۱۰۰۰ نقطه اخیر
        if len(self.path_msg.poses) > 1000:
            self.path_msg.poses.pop(0)
            
        self.path_pub.publish(self.path_msg)

        # ۲. ساخت مارکر نقطه‌ای خودرو (یک کره در مرکز base_link)
        car_marker = Marker()
        car_marker.header.stamp = msg.header.stamp
        car_marker.header.frame_id = 'base_link'  # فریم محلی خودرو
        car_marker.ns = 'vehicle_model'
        car_marker.id = 0
        car_marker.type = Marker.SPHERE
        car_marker.action = Marker.ADD

        # موقعیت نقطه در مرکز بدنه
        car_marker.pose.position.x = 0.0
        car_marker.pose.position.y = 0.0
        car_marker.pose.position.z = 0.0
        car_marker.pose.orientation.w = 1.0

        # ابعاد نقطه (کره با قطر ۰.۲ متر)
        car_marker.scale.x = 0.2
        car_marker.scale.y = 0.2
        car_marker.scale.z = 0.2

        # رنگ نقطه (قرمز شفاف)
        car_marker.color.r = 1.0
        car_marker.color.g = 0.0
        car_marker.color.b = 0.0
        car_marker.color.a = 1.0

        self.marker_pub.publish(car_marker)

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