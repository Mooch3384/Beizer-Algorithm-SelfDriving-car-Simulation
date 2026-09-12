import numpy as np
import cv2
from typing import Tuple, Dict, Any, List, Optional


class EgoLaneDetector:
    """
    Ego-Lane Segmentation and Local Mapping module.
    Isolates vehicle ego-lane using color thresholding, column histogram peaks,
    sliding window pixel extraction, and 2nd-order polynomial curve fitting.
    """

    def __init__(
        self,
        nwindows: int = 10,
        margin: int = 40,
        minpix: int = 50,
        meters_per_pixel_x: float = 3.7 / 200.0,  # 3.7m standard lane width in ~200 pixels
        meters_per_pixel_y: float = 10.0 / 480.0  # 10m longitudinal distance in 480 pixels
    ):
        self.nwindows = nwindows
        self.margin = margin
        self.minpix = minpix
        self.meters_per_pixel_x = meters_per_pixel_x
        self.meters_per_pixel_y = meters_per_pixel_y

    def binarize(self, bev_img: np.ndarray) -> np.ndarray:
        """
        Apply color space thresholding and Sobel gradient filtering to isolate lane lines.

        :param bev_img: Top-down BEV image (BGR).
        :return: Binary image (255 for lane pixels, 0 for background).
        """
        if len(bev_img.shape) == 2:
            bev_bgr = cv2.cvtColor(bev_img, cv2.COLOR_GRAY2BGR)
        else:
            bev_bgr = bev_img.copy()

        hls = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]

        # 1. White color thresholding (High Luminance)
        white_binary = np.zeros_like(l_channel)
        white_binary[(l_channel > 180)] = 255

        # 2. Yellow color thresholding (High Saturation & specific Hue)
        hsv = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([15, 80, 80], dtype=np.uint8)
        upper_yellow = np.array([35, 255, 255], dtype=np.uint8)
        yellow_binary = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # 3. Sobel X Gradient thresholding
        gray = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobelx = np.absolute(sobelx)
        max_val = np.max(abs_sobelx)
        if max_val > 0:
            scaled_sobel = np.uint8(255 * abs_sobelx / max_val)
        else:
            scaled_sobel = np.uint8(abs_sobelx)

        sobel_binary = np.zeros_like(scaled_sobel)
        sobel_binary[(scaled_sobel >= 30) & (scaled_sobel <= 255)] = 255

        # Combined binary mask
        combined_binary = np.zeros_like(gray)
        combined_binary[(white_binary == 255) | (yellow_binary == 255) | (sobel_binary == 255)] = 255

        return combined_binary

    def detect_lane(self, binary_bev: np.ndarray) -> Dict[str, Any]:
        """
        Perform sliding window search and 2nd-order polynomial fitting.

        :param binary_bev: Binary top-down BEV image.
        :return: Dictionary containing polynomial coefficients, pixel coordinates, centerline, and debug image.
        """
        img_h, img_w = binary_bev.shape
        histogram = np.sum(binary_bev[img_h // 2:, :], axis=0)

        midpoint = img_w // 2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Fallbacks if base peaks are invalid
        if histogram[leftx_base] < 100:
            leftx_base = img_w // 4
        if histogram[rightx_base] < 100:
            rightx_base = 3 * img_w // 4

        window_height = img_h // self.nwindows
        nonzero = binary_bev.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        debug_img = cv2.cvtColor(binary_bev, cv2.COLOR_GRAY2BGR)

        for window in range(self.nwindows):
            win_y_low = img_h - (window + 1) * window_height
            win_y_high = img_h - window * window_height
            
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin

            # Draw search windows on debug image
            cv2.rectangle(debug_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(debug_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

            good_left_inds = (
                (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)
            ).nonzero()[0]
            
            good_right_inds = (
                (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)
            ).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > self.minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        ploty = np.linspace(0, img_h - 1, img_h)

        left_fit = None
        right_fit = None
        center_fit = None

        if len(leftx) > self.minpix:
            left_fit = np.polyfit(lefty, leftx, 2)
            left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
            debug_img[lefty, leftx] = [0, 0, 255]  # Red for left line
        else:
            left_fitx = ploty * 0 + (img_w // 4)

        if len(rightx) > self.minpix:
            right_fit = np.polyfit(righty, rightx, 2)
            right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
            debug_img[righty, rightx] = [255, 0, 0]  # Blue for right line
        else:
            right_fitx = ploty * 0 + (3 * img_w // 4)

        # Centerline computation
        if left_fit is not None and right_fit is not None:
            center_fitx = (left_fitx + right_fitx) / 2.0
            center_fit = np.polyfit(ploty, center_fitx, 2)
        elif left_fit is not None:
            # Shift left line by estimated lane width (~200 pixels)
            center_fitx = left_fitx + 100.0
            center_fit = np.polyfit(ploty, center_fitx, 2)
        elif right_fit is not None:
            center_fitx = right_fitx - 100.0
            center_fit = np.polyfit(ploty, center_fitx, 2)
        else:
            center_fitx = ploty * 0 + (img_w // 2)
            center_fit = np.array([0.0, 0.0, float(img_w // 2)])

        # Draw fitted polynomial curves on BEV debug frame
        for i in range(len(ploty) - 1):
            y1, y2 = int(ploty[i]), int(ploty[i + 1])
            # Left line
            if 0 <= left_fitx[i] < img_w and 0 <= left_fitx[i + 1] < img_w:
                cv2.line(debug_img, (int(left_fitx[i]), y1), (int(left_fitx[i + 1]), y2), (0, 255, 255), 2)
            # Right line
            if 0 <= right_fitx[i] < img_w and 0 <= right_fitx[i + 1] < img_w:
                cv2.line(debug_img, (int(right_fitx[i]), y1), (int(right_fitx[i + 1]), y2), (255, 255, 0), 2)
            # Centerline
            if 0 <= center_fitx[i] < img_w and 0 <= center_fitx[i + 1] < img_w:
                cv2.line(debug_img, (int(center_fitx[i]), y1), (int(center_fitx[i + 1]), y2), (0, 255, 0), 3)

        return {
            "left_fit": left_fit,
            "right_fit": right_fit,
            "center_fit": center_fit,
            "ploty": ploty,
            "left_fitx": left_fitx,
            "right_fitx": right_fitx,
            "center_fitx": center_fitx,
            "debug_img": debug_img
        }

    def bev_to_base_link(
        self,
        pixel_x: np.ndarray,
        pixel_y: np.ndarray,
        img_shape: Tuple[int, int]
    ) -> List[Tuple[float, float]]:
        """
        Convert BEV image pixel coordinates to vehicle base_link frame in meters.
        base_link standard frame: X is forward, Y is left, Z is up.
        Origin (0,0) is at bottom center of BEV image (x = img_w / 2, y = img_h).

        :param pixel_x: Array of x pixel coordinates in BEV image.
        :param pixel_y: Array of y pixel coordinates in BEV image.
        :param img_shape: (height, width) of BEV image.
        :return: List of (x_meters, y_meters) in base_link frame.
        """
        img_h, img_w = img_shape
        center_x = img_w / 2.0

        coords_base_link = []
        for px, py in zip(pixel_x, pixel_y):
            # Forward distance X (meters)
            x_m = (img_h - py) * self.meters_per_pixel_y
            # Lateral distance Y (meters, left is positive in ROS)
            y_m = (center_x - px) * self.meters_per_pixel_x
            coords_base_link.append((x_m, y_m))

        return coords_base_link
