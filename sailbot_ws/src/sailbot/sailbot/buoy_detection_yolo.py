"""
YOLO buoy detection node for Sailbot.

This module provides a ROS2 node that runs TensorRT-backed YOLO buoy detection and
publishes geographic buoy locations in the same format/channels used by the original
blob-detection pipeline.
"""

import os
import time
from typing import List

import cv2
import numpy as np
import pyzed.sl as sl
import rclpy
from rclpy.node import Node
from scipy.optimize import linear_sum_assignment
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float64
from std_msgs.msg import Header

from sailbot.buoy_detection import (
    CAMERA_RESOLUTION,
    CX,
    CY,
    FX,
    FY,
    Track,
    calculate_offset_position,
    enu_to_geodetic,
    geodetic_to_enu,
)
from sailbot_msgs.msg import BuoyDetectionStamped


# -----------------------------------------------------------------------------
# Structural Integration Notes (comments only, non-functional)
# -----------------------------------------------------------------------------
# This file intentionally preserves the previous architecture's public outputs
# while replacing the internals of detection and frame acquisition:
# - The old node used ROS image/depth topics + blob/HSV segmentation.
# - This node uses direct ZED SDK capture (`pyzed.sl`) + YOLO TensorRT inference.
# - The old `cv_mask` debug-video output is intentionally disabled here; YOLO
#   recognition only needs detections from model inference and `buoy_position`.
#
# Known working command sequence used during integration/testing:
# 1) cd /home/sailbot/sailbot24-25
# 2) source .venv/bin/activate
# 3) source /opt/ros/humble/setup.bash
# 4) source /home/sailbot/sailbot24-25/sailbot_ws/install/setup.bash
# 5) (if TensorRT Python bindings are from system dist-packages) [dont do this]
#    export PYTHONPATH="/usr/lib/python3/dist-packages:/usr/lib/python3.10/dist-packages:$PYTHONPATH"
# 6) python /home/sailbot/sailbot24-25/sailbot_ws/src/sailbot/sailbot/buoy_detection_yolo.py \      --ros-args --params-file /home/sailbot/sailbot24-25/sailbot_ws/src/sailbot/config/config.yaml
# 7) In a second sourced terminal:
#    ros2 topic echo /buoy_position
# -----------------------------------------------------------------------------


class BuoyDetectionYOLO(Node):
    """
    A ROS2 node responsible for detecting buoys using a YOLO TensorRT engine and
    publishing geographic buoy detections.

    This node preserves the external ROS interfaces from the original buoy
    detector where practical, while changing the detection backend from
    blob/HSV logic to neural network inference.

    :ivar current_x_scaling_factor: Scaling factor to adjust the pixel coordinates based on image width.
    :ivar current_y_scaling_factor: Scaling factor to adjust the pixel coordinates based on image height.
    :ivar latitude: Current latitude of the vessel.
    :ivar longitude: Current longitude of the vessel.
    :ivar heading: Current heading of the vessel in degrees.
    :ivar tracks: List of active tracking objects representing detected buoys.

    **Methods**:
    - **publish_tracks**: Publishes tracked buoy positions on `buoy_position`.
    - **associate_detections_to_tracks**: Matches frame detections to existing tracks.
    - **capture_and_process**: Captures ZED frames, runs YOLO, projects detections, updates tracks, and publishes output.
    - **pixel_to_world**: Converts pixel coordinates to world coordinates based on camera intrinsics and depth.

    **Usage**:
    - Node can be launched as a drop-in replacement for buoy detection in ROS graph integration.

    **Notes**:
    - This node uses direct ZED SDK frame capture and does not require ROS image topics.
    """

    current_x_scaling_factor = 1.0
    current_y_scaling_factor = 1.0
    latitude, longitude = 42.84456, -70.97622
    heading = 0.0
    tracks: List[Track] = []

    # Tune these here to control chronological filtering before track processing.
    MIN_RECOGNITION_SECONDS = 2.0
    MIN_RECOGNITION_FRAME_RATIO = 0.5

    def __init__(self):
        super().__init__("object_detection_node")

        self.recognition_count_history: list[tuple[float, int]] = []
        self.recognition_filter_start_time: float | None = None
        self.recognition_gate_open = False

        self.set_parameters()
        self.get_parameters()
        self._load_yolo_model()
        self._open_zed()

        self.airmar_position_subscription = self.create_subscription(
            NavSatFix,
            "/airmar_data/lat_long",
            self.airmar_position_callback,
            10,
        )
        self.airmar_heading_subscription = self.create_subscription(
            Float64,
            "heading",
            self.airmar_heading_callback,
            10,
        )

        self.buoy_position_publisher = self.create_publisher(BuoyDetectionStamped, "buoy_position", 10)

        self.process_timer = self.create_timer(self.capture_period_seconds, self.capture_and_process)
        self.get_logger().info("YOLO buoy detection setup complete")

    def set_parameters(self) -> None:
        self.declare_parameter("sailbot.cv.buoy_detection_lifetime_seconds", 3.0)

        self.declare_parameter(
            "sailbot.cv.yolo.engine_path",
            "/home/sailbot/Downloads/yolobuoyV2-engine-validation/my_model_fp16.engine",
        )
        self.declare_parameter("sailbot.cv.yolo.conf_threshold", 0.25)
        self.declare_parameter("sailbot.cv.yolo.iou_threshold", 0.45)
        self.declare_parameter("sailbot.cv.yolo.imgsz", 640)
        self.declare_parameter("sailbot.cv.yolo.device", "0")
        self.declare_parameter("sailbot.cv.yolo.max_detection_distance_meters", 120.0)
        self.declare_parameter("sailbot.cv.yolo.max_association_distance_meters", 3.0)
        self.declare_parameter("sailbot.cv.yolo.capture_period_seconds", 0.05)
        self.declare_parameter("sailbot.cv.yolo.zed_resolution", "VGA")
        self.declare_parameter("sailbot.cv.yolo.zed_depth_mode", "NEURAL")

    def get_parameters(self) -> None:
        self.buoy_detection_lifetime_seconds = (
            self.get_parameter("sailbot.cv.buoy_detection_lifetime_seconds").get_parameter_value().double_value
        )

        self.yolo_engine_path = self.get_parameter("sailbot.cv.yolo.engine_path").get_parameter_value().string_value
        self.yolo_conf_threshold = self.get_parameter("sailbot.cv.yolo.conf_threshold").get_parameter_value().double_value
        self.yolo_iou_threshold = self.get_parameter("sailbot.cv.yolo.iou_threshold").get_parameter_value().double_value
        self.yolo_imgsz = self.get_parameter("sailbot.cv.yolo.imgsz").get_parameter_value().integer_value
        self.yolo_device = self.get_parameter("sailbot.cv.yolo.device").get_parameter_value().string_value
        self.max_detection_distance_meters = (
            self.get_parameter("sailbot.cv.yolo.max_detection_distance_meters").get_parameter_value().double_value
        )
        self.max_association_distance_meters = (
            self.get_parameter("sailbot.cv.yolo.max_association_distance_meters").get_parameter_value().double_value
        )
        self.capture_period_seconds = (
            self.get_parameter("sailbot.cv.yolo.capture_period_seconds").get_parameter_value().double_value
        )
        self.zed_resolution_name = self.get_parameter("sailbot.cv.yolo.zed_resolution").get_parameter_value().string_value
        self.zed_depth_mode_name = self.get_parameter("sailbot.cv.yolo.zed_depth_mode").get_parameter_value().string_value

    def _load_yolo_model(self) -> None:
        if not os.path.exists(self.yolo_engine_path):
            raise FileNotFoundError(
                f"YOLO engine not found at '{self.yolo_engine_path}'. "
                "Set sailbot.cv.yolo.engine_path to a valid TensorRT engine."
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is required for buoy_detection_yolo. "
                "Install ultralytics in the runtime environment."
            ) from exc
        self.model = YOLO(self.yolo_engine_path, task="detect")
        self.get_logger().info(f"Loaded YOLO engine: {self.yolo_engine_path}")

    def _open_zed(self) -> None:
        resolution_map = {
            "VGA": sl.RESOLUTION.VGA,
            "HD720": sl.RESOLUTION.HD720,
            "HD1080": sl.RESOLUTION.HD1080,
            "HD2K": sl.RESOLUTION.HD2K,
        }
        depth_map = {
            "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE,
            "QUALITY": sl.DEPTH_MODE.QUALITY,
            "ULTRA": sl.DEPTH_MODE.ULTRA,
            "NEURAL": sl.DEPTH_MODE.NEURAL,
        }
        resolution = resolution_map.get(self.zed_resolution_name.upper(), sl.RESOLUTION.VGA)
        depth_mode = depth_map.get(self.zed_depth_mode_name.upper(), sl.DEPTH_MODE.NEURAL)

        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = resolution
        init_params.depth_mode = depth_mode
        init_params.coordinate_units = sl.UNIT.METER

        status = self.zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED open failed: {status}")

        self.runtime_params = sl.RuntimeParameters()
        self.runtime_params.enable_depth = True
        self.image_zed = sl.Mat()
        self.depth_zed = sl.Mat()
        self.get_logger().info(
            f"Opened ZED with resolution={self.zed_resolution_name}, depth_mode={self.zed_depth_mode_name}"
        )

    def airmar_position_callback(self, msg: NavSatFix) -> None:
        self.latitude = msg.latitude
        self.longitude = msg.longitude

    def airmar_heading_callback(self, msg: Float64) -> None:
        self.heading = msg.data

    def remove_stale_tracks(self) -> None:
        current_time = time.time()
        self.tracks = [
            track for track in self.tracks if current_time - track.last_update_time <= self.buoy_detection_lifetime_seconds
        ]

    def publish_tracks(self, header: Header) -> None:
        for track in self.tracks:
            if track.time_since_update != 0:
                continue
            enu_position = track.get_position()
            lat, lon = enu_to_geodetic(enu_position[0], enu_position[1])
            detection = BuoyDetectionStamped()
            detection.header = header
            detection.position.latitude = lat
            detection.position.longitude = lon
            detection.id = track.id
            self.buoy_position_publisher.publish(detection)

    def associate_detections_to_tracks(self, tracks, detections_enu, max_distance: float):
        if len(tracks) == 0:
            return [], set(range(len(detections_enu)))
        if len(detections_enu) == 0:
            return [], []

        cost_matrix = np.zeros((len(tracks), len(detections_enu)), dtype=np.float32)
        for t, track in enumerate(tracks):
            predicted_enu = track.predict()
            for d, detection_enu in enumerate(detections_enu):
                cost_matrix[t, d] = np.linalg.norm(np.array(predicted_enu) - np.array(detection_enu))

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches = []
        unmatched_detections = set(range(len(detections_enu)))
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < max_distance:
                matches.append((r, c))
                unmatched_detections.remove(c)
        return matches, list(unmatched_detections)

    def _get_valid_depth(self, cx: int, cy: int) -> float | None:
        depth_err, depth = self.depth_zed.get_value(cx, cy)
        if depth_err != sl.ERROR_CODE.SUCCESS:
            return None
        depth_value = float(depth)
        if np.isnan(depth_value) or depth_value <= 0.0 or not np.isfinite(depth_value):
            return None
        if depth_value > self.max_detection_distance_meters:
            return None
        return depth_value

    def _recognition_gate_is_open(self, valid_detection_count: int) -> bool:
        current_time = time.time()
        if self.recognition_filter_start_time is None:
            self.recognition_filter_start_time = current_time

        self.recognition_count_history.append((current_time, valid_detection_count))

        window_start = current_time - self.MIN_RECOGNITION_SECONDS
        self.recognition_count_history = [
            entry for entry in self.recognition_count_history if entry[0] >= window_start
        ]

        gate_open = False
        gate_ratio = 0.0
        expected_count = 0
        window_is_mature = current_time - self.recognition_filter_start_time >= self.MIN_RECOGNITION_SECONDS

        if window_is_mature:
            nonzero_counts = [count for _, count in self.recognition_count_history if count > 0]
            if nonzero_counts:
                expected_count = max(1, int(np.median(nonzero_counts)))
                recognized_frames = sum(
                    1 for _, count in self.recognition_count_history if count >= expected_count
                )
                gate_ratio = recognized_frames / len(self.recognition_count_history)
                gate_open = gate_ratio >= self.MIN_RECOGNITION_FRAME_RATIO

        if gate_open != self.recognition_gate_open:
            self.recognition_gate_open = gate_open
            if gate_open:
                self.get_logger().info(
                    "YOLO recognition gate opened "
                    f"(count>={expected_count}, uptime={gate_ratio:.2f})"
                )
            else:
                self.get_logger().info(
                    "YOLO recognition gate closed "
                    f"(count>={expected_count}, uptime={gate_ratio:.2f})"
                )

        return gate_open and valid_detection_count >= expected_count

    def pixel_to_world(self, x_pixel: int, y_pixel: int, depth: float, f_x: float, f_y: float, c_x: float, c_y: float):
        x_norm = (x_pixel - c_x) / f_x
        y_norm = (y_pixel - c_y) / f_y
        x_world = x_norm * depth
        y_world = y_norm * depth
        return x_world, y_world, depth

    def capture_and_process(self) -> None:
        if self.zed.grab(self.runtime_params) != sl.ERROR_CODE.SUCCESS:
            return

        self.zed.retrieve_image(self.image_zed, sl.VIEW.LEFT, sl.MEM.CPU)
        self.zed.retrieve_measure(self.depth_zed, sl.MEASURE.DEPTH, sl.MEM.CPU)

        frame = self.image_zed.get_data()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        self.current_x_scaling_factor = frame_bgr.shape[1] / CAMERA_RESOLUTION[0]
        self.current_y_scaling_factor = frame_bgr.shape[0] / CAMERA_RESOLUTION[1]

        results = self.model.predict(
            frame_bgr,
            conf=self.yolo_conf_threshold,
            iou=self.yolo_iou_threshold,
            imgsz=self.yolo_imgsz,
            device=self.yolo_device,
            verbose=False,
        )

        detections_enu = []
        boxes = results[0].boxes if results else None

        if boxes is not None and boxes.xyxy is not None and len(boxes.xyxy) > 0:
            xyxy = boxes.xyxy.cpu().numpy()

            for bbox in xyxy:
                x1, y1, x2, y2 = bbox.tolist()
                cx = int((x1 + x2) / 2.0)
                cy = int((y1 + y2) / 2.0)
                depth = self._get_valid_depth(cx, cy)
                if depth is None:
                    continue

                world_coords = self.pixel_to_world(
                    cx,
                    cy,
                    depth,
                    FX * self.current_x_scaling_factor,
                    FY * self.current_y_scaling_factor,
                    CX * self.current_x_scaling_factor,
                    CY * self.current_y_scaling_factor,
                )
                latlon = calculate_offset_position(self.latitude, self.longitude, self.heading, world_coords[2], world_coords[0])
                detections_enu.append(geodetic_to_enu(latlon.latitude, latlon.longitude))

        if not self._recognition_gate_is_open(len(detections_enu)):
            self.remove_stale_tracks()
            return

        matches, unmatched = self.associate_detections_to_tracks(
            self.tracks, detections_enu, self.max_association_distance_meters
        )
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(detections_enu[det_idx])
        for det_idx in unmatched:
            enu = detections_enu[det_idx]
            self.tracks.append(Track(enu[0], enu[1]))

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "zed_left_camera"
        self.publish_tracks(header)
        self.remove_stale_tracks()

    def destroy_node(self):
        if hasattr(self, "zed") and self.zed is not None:
            self.zed.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BuoyDetectionYOLO()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()