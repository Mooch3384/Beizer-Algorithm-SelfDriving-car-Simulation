#!/usr/bin/env python3
"""
Simulator Image Filter Node
  Sub: /camera/image_raw (sensor_msgs/msg/Image) — 512x512 from simulator
  Pub: /filtered_image   (sensor_msgs/msg/Image, mono8) — binary warped lane mask
  Pub: /filtered_debug   (sensor_msgs/msg/Image, bgr8)  — visual color overlay
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class ImageFilterNode(Node):
    def __init__(self):
        super().__init__('image_filter_node')

        # Real-time QoS matching simulator stream
        rt_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Dynamic parameter declarations
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('tl_x', 160)
        self.declare_parameter('tl_y', 260)
        self.declare_parameter('tr_x', 352)
        self.declare_parameter('tr_y', 260)
        self.declare_parameter('bl_x', 20)
        self.declare_parameter('bl_y', 480)
        self.declare_parameter('br_x', 492)
        self.declare_parameter('br_y', 480)
        self.declare_parameter('out_width', 512)
        self.declare_parameter('out_height', 300)

        self.declare_parameter('mask_hood', True)
        self.declare_parameter('hood_ymin', 400)
        self.declare_parameter('hood_ymax', 512)
        self.declare_parameter('hood_xmin', 130)
        self.declare_parameter('hood_xmax', 380)

        self.declare_parameter('yellow_h_min', 15)
        self.declare_parameter('yellow_h_max', 38)
        self.declare_parameter('yellow_s_min', 60)
        self.declare_parameter('yellow_s_max', 255)
        self.declare_parameter('yellow_v_min', 80)
        self.declare_parameter('yellow_v_max', 255)

        self.declare_parameter('white_s_max', 45)
        self.declare_parameter('white_v_min', 180)

        self.declare_parameter('use_sobel', False)
        self.declare_parameter('sobel_thresh', 45)
        self.declare_parameter('blur_ksize', 5)
        self.declare_parameter('morph_ksize', 3)

        self.br = CvBridge()
        self._load_parameters()
        self._calc_perspective()

        # Check genuine CUDA support
        try:
            if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.gpu_frame = cv2.cuda_GpuMat()
                self.use_gpu = True
            else:
                self.use_gpu = False
        except Exception:
            self.use_gpu = False

        # Subscribers & Publishers
        camera_topic = self.get_parameter('camera_topic').value
        self.subscription = self.create_subscription(
            Image, camera_topic, self.listener_callback, rt_qos)
        self.filtered_pub = self.create_publisher(Image, '/filtered_image', rt_qos)
        self.debug_pub = self.create_publisher(Image, '/filtered_debug', rt_qos)

        self.add_on_set_parameters_callback(self._param_callback)
        self.frame_count = 0
        self.get_logger().info(f'ImageFilterNode ready (GPU={self.use_gpu}) on topic: {camera_topic}')

    def _load_parameters(self):
        p = self.get_parameter
        self.tl = (p('tl_x').value, p('tl_y').value)
        self.tr = (p('tr_x').value, p('tr_y').value)
        self.bl = (p('bl_x').value, p('bl_y').value)
        self.br_pt = (p('br_x').value, p('br_y').value)
        self.out_width = p('out_width').value
        self.out_height = p('out_height').value

        self.mask_hood = p('mask_hood').value
        self.hood_ymin = p('hood_ymin').value
        self.hood_ymax = p('hood_ymax').value
        self.hood_xmin = p('hood_xmin').value
        self.hood_xmax = p('hood_xmax').value

        self.yellow_lower = np.array([p('yellow_h_min').value, p('yellow_s_min').value, p('yellow_v_min').value], dtype=np.uint8)
        self.yellow_upper = np.array([p('yellow_h_max').value, p('yellow_s_max').value, p('yellow_v_max').value], dtype=np.uint8)

        self.white_lower = np.array([0, 0, p('white_v_min').value], dtype=np.uint8)
        self.white_upper = np.array([180, p('white_s_max').value, 255], dtype=np.uint8)

        self.use_sobel = p('use_sobel').value
        self.sobel_thresh = p('sobel_thresh').value
        self.blur_ksize = p('blur_ksize').value
        self.morph_ksize = p('morph_ksize').value

        k = self.morph_ksize
        if k > 1 and k % 2 == 1:
            self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        else:
            self.morph_kernel = None

    def _param_callback(self, params):
        for param in params:
            self.get_logger().info(f'Parameter updated: {param.name} = {param.value}')
        self._load_parameters()
        self._calc_perspective()
        return SetParametersResult(successful=True)

    def _calc_perspective(self):
        pts = np.float32([self.tl, self.tr, self.bl, self.br_pt])
        dst = np.float32([
            [0, 0],
            [self.out_width - 1, 0],
            [0, self.out_height - 1],
            [self.out_width - 1, self.out_height - 1]
        ])
        self.M = cv2.getPerspectiveTransform(pts, dst)

    def filter_lanes(self, warped_bgr):
        hsv = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2HSV)

        # Yellow line extraction (left lane)
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)

        # White line extraction (right lane)
        white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)

        # Combined color mask
        binary = cv2.bitwise_or(yellow_mask, white_mask)

        # Optional Sobel-X gradient reinforcement
        if self.use_sobel:
            gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
            if self.blur_ksize > 1:
                bk = self.blur_ksize if self.blur_ksize % 2 == 1 else self.blur_ksize + 1
                gray = cv2.GaussianBlur(gray, (bk, bk), 0)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            abs_s = np.absolute(sobelx)
            scaled = np.uint8(255 * abs_s / (np.max(abs_s) + 1e-6))
            sobel_mask = (scaled >= self.sobel_thresh).astype(np.uint8) * 255
            binary = cv2.bitwise_or(binary, sobel_mask)

        # Morphological noise removal and gap bridging
        if self.morph_kernel is not None:
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.morph_kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.morph_kernel)

        return binary, yellow_mask, white_mask

    def listener_callback(self, msg):
        try:
            frame = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Mask out vehicle hood before warping
            if self.mask_hood:
                hood_poly = np.array([
                    [self.hood_xmin - 10, self.hood_ymax],
                    [self.hood_xmin + 10, self.hood_ymin],
                    [self.hood_xmax - 10, self.hood_ymin],
                    [self.hood_xmax + 10, self.hood_ymax]
                ], dtype=np.int32)
                cv2.fillPoly(frame, [hood_poly], (0, 0, 0))

            # Bird's Eye View perspective warp
            if self.use_gpu:
                self.gpu_frame.upload(frame)
                gw = cv2.cuda.warpPerspective(self.gpu_frame, self.M, (self.out_width, self.out_height))
                warped = gw.download()
            else:
                warped = cv2.warpPerspective(frame, self.M, (self.out_width, self.out_height))

            # Segmentation
            binary, yellow_m, white_m = self.filter_lanes(warped)

            # Publish filtered binary mask
            out = self.br.cv2_to_imgmsg(binary, encoding='mono8')
            out.header = msg.header
            self.filtered_pub.publish(out)

            # Publish debug visualizer every 3 frames
            self.frame_count += 1
            if self.frame_count % 3 == 0:
                debug_bgr = warped.copy()
                # Highlight yellow line in yellow and white line in green
                debug_bgr[yellow_m > 0] = [0, 255, 255]
                debug_bgr[white_m > 0] = [0, 255, 0]
                dbg_msg = self.br.cv2_to_imgmsg(debug_bgr, encoding='bgr8')
                dbg_msg.header = msg.header
                self.debug_pub.publish(dbg_msg)

        except Exception as e:
            self.get_logger().error(f'ImageFilterNode Error: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = ImageFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
