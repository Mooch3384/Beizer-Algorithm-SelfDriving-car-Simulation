import numpy as np
import cv2
from typing import List, Tuple, Optional


class IPMTransformer:
    """
    Inverse Perspective Mapping (IPM) Transformer module.
    Transforms perspective front-camera images to top-down Bird's-Eye View (BEV).
    """

    def __init__(
        self,
        src_points: Optional[List[List[float]]] = None,
        dst_points: Optional[List[List[float]]] = None,
        out_shape: Tuple[int, int] = (640, 480)
    ):
        """
        Initialize IPM Homography matrices.

        :param src_points: 4 source points in camera perspective [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
        :param dst_points: 4 destination points in top-down BEV perspective
        :param out_shape: Target BEV image dimensions (width, height)
        """
        self.width, self.height = out_shape

        # Default source trapezoid points if not provided (assume 640x480 resolution)
        if src_points is None:
            src_points = [
                [180, 330],  # Top-Left
                [460, 330],  # Top-Right
                [610, 470],  # Bottom-Right
                [30, 470]    # Bottom-Left
            ]

        if dst_points is None:
            margin = 100
            dst_points = [
                [margin, 0],             # Top-Left
                [self.width - margin, 0], # Top-Right
                [self.width - margin, self.height], # Bottom-Right
                [margin, self.height]     # Bottom-Left
            ]

        self.src_points = np.float32(src_points)
        self.dst_points = np.float32(dst_points)

        self.M = None
        self.M_inv = None
        self._compute_matrices()

    def _compute_matrices(self) -> None:
        """Compute forward and inverse perspective transformation matrices."""
        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.M_inv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)

    def update_points(
        self,
        src_points: List[List[float]],
        dst_points: List[List[float]],
        out_shape: Optional[Tuple[int, int]] = None
    ) -> None:
        """Update source and destination calibration matrices dynamically at runtime."""
        if out_shape is not None:
            self.width, self.height = out_shape
        self.src_points = np.float32(src_points)
        self.dst_points = np.float32(dst_points)
        self._compute_matrices()

    def transform(self, image: np.ndarray) -> np.ndarray:
        """
        Warp perspective camera image to top-down BEV image.

        :param image: Input OpenCV BGR/RGB or grayscale image.
        :return: Warped BEV image.
        """
        if image is None or image.size == 0:
            raise ValueError("Input image to IPM transform is empty or None")

        return cv2.warpPerspective(
            image,
            self.M,
            (self.width, self.height),
            flags=cv2.INTER_LINEAR
        )

    def unproject_bev_points(self, bev_points: np.ndarray) -> np.ndarray:
        """
        Transform 2D points from BEV pixel space to camera perspective pixel space.

        :param bev_points: Array of shape (N, 2) [x_bev, y_bev]
        :return: Array of shape (N, 2) [x_cam, y_cam]
        """
        if len(bev_points) == 0:
            return np.empty((0, 2), dtype=np.float32)

        pts = np.float32(bev_points).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, self.M_inv)
        return transformed.reshape(-1, 2)
