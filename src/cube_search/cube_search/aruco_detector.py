"""Node aruco_detector: obraz z kamery -> pozycje kostek w ramce globalnej (map).

Lancuch transformacji dla kazdego markera:
    T^map_kostka = T^map_odom(AMCL, TF; korekta dryfu odometrii)
                  * T^odom_base(/odom, interpolowane na stemplu obrazu)
                  * T^base_cam(tf_static z URDF)
                  * R_cam_opt (camera_rgb_frame -> ramka optyczna)
                  * T^opt_marker(solvePnP IPPE_SQUARE)
                  * przesuniecie o pol boku kostki w glab (srodek kostki)
Publikowane jest kazde przetworzone zdjecie (takze bez detekcji), bo poza kamery z
wiadomosci sluzy serwerowi akcji do aktualizacji mapy pokrycia.
"""
import time
from collections import defaultdict

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener

from cube_search_interfaces.msg import CubeDetection, CubeDetectionArray

from .aruco import DEFAULT_MARKER_LENGTH, ArucoDetector
from .geometry import (R_CAM_OPT, OdomBuffer, cube_center_from_marker, quat_from_rot,
                       rot_from_quat, rot_z, yaw_from_quat)


class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__('aruco_detector')
        p = self.declare_parameter
        p('image_topic', '/camera/image_raw')
        p('camera_info_topic', '/camera/camera_info')
        p('odom_topic', '/odom')
        p('detections_topic', '/cube_search/detections')
        p('debug_image_topic', '/cube_search/debug_image')
        p('base_frame', 'base_footprint')
        p('camera_frame', 'camera_rgb_frame')
        p('marker_length', DEFAULT_MARKER_LENGTH)
        p('cube_size', 0.09)
        p('dictionary', 'DICT_ARUCO_ORIGINAL')
        p('process_rate', 5.0)
        p('max_odom_extrapolation', 0.05)
        p('min_marker_px', 12.0)
        p('max_reproj_err', 3.0)
        p('debug_image_scale', 0.5)
        p('extrinsics_timeout', 10.0)
        p('camera_offset_fallback', [0.069, -0.047, 0.117])
        p('global_frame', 'map')  # 'odom' = bez korekty AMCL
        g = lambda name: self.get_parameter(name).value  # noqa: E731

        self.detector = ArucoDetector(g('marker_length'), g('dictionary'),
                                      g('min_marker_px'), g('max_reproj_err'))
        self.half = 0.5 * float(g('cube_size'))
        self.max_extra = float(g('max_odom_extrapolation'))
        self.base_frame = g('base_frame')
        self.camera_frame = g('camera_frame')
        self.debug_scale = float(g('debug_image_scale'))
        self.extrinsics_timeout = float(g('extrinsics_timeout'))
        self.fallback_offset = np.array(g('camera_offset_fallback'), dtype=float)
        self.global_frame = g('global_frame')
        self._tf_fail = 0

        self.bridge = CvBridge()
        self.odom = OdomBuffer(max_age=3.0)
        self.odom_frame = 'odom'
        self.K = None
        self.D = None
        self.R_base_cam = None
        self.t_base_cam = None
        self._image = None
        self._last_stamp = None
        self._t0 = time.monotonic()
        self._frames = 0
        self._det_count = defaultdict(int)
        self._max_dist = defaultdict(float)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        img_qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                             reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, g('image_topic'), self._on_image, img_qos)
        self.create_subscription(CameraInfo, g('camera_info_topic'), self._on_info, 1)
        self.create_subscription(Odometry, g('odom_topic'), self._on_odom, 50)
        self.pub = self.create_publisher(CubeDetectionArray, g('detections_topic'), 10)
        self.dbg_pub = self.create_publisher(Image, g('debug_image_topic'), 1)
        self.create_timer(1.0 / float(g('process_rate')), self._on_timer)
        self.create_timer(20.0, self._log_stats)
        self.get_logger().info(
            f'aruco_detector: slownik {g("dictionary")}, bok markera {g("marker_length"):.4f} m, '
            f'{g("process_rate"):.1f} Hz')

    # ----------------------------------------------------------- callbacks
    def _on_image(self, msg):
        self._image = msg

    def _on_info(self, msg):
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.D = np.array(msg.d, dtype=np.float64) if len(msg.d) else np.zeros(5)

    def _on_odom(self, msg):
        self.odom_frame = msg.header.frame_id or 'odom'
        q = msg.pose.pose.orientation
        self.odom.add(Time.from_msg(msg.header.stamp).nanoseconds * 1e-9,
                      msg.pose.pose.position.x, msg.pose.pose.position.y,
                      yaw_from_quat(q.x, q.y, q.z, q.w))

    def _lookup_extrinsics(self):
        """Przeksztalcenie kamery wzgledem bazy robota (tf_static z robot_state_publisher)."""
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, self.camera_frame, Time())
        except Exception as exc:  # noqa: BLE001
            if time.monotonic() - self._t0 < self.extrinsics_timeout:
                return False
            self.t_base_cam = self.fallback_offset.copy()
            self.R_base_cam = np.eye(3)
            self.get_logger().warn(f'Brak TF {self.base_frame}->{self.camera_frame} ({exc}); '
                                   f'uzywam stalej z URDF {self.t_base_cam.tolist()}')
            return True
        t, q = tf.transform.translation, tf.transform.rotation
        self.t_base_cam = np.array([t.x, t.y, t.z])
        self.R_base_cam = rot_from_quat(q.x, q.y, q.z, q.w)
        self.get_logger().info(f'T_base_cam: t={np.round(self.t_base_cam, 4).tolist()}, '
                               f'q=({q.x:.3f}, {q.y:.3f}, {q.z:.3f}, {q.w:.3f})')
        return True

    def _on_timer(self):
        msg = self._image
        if msg is None or self.K is None:
            return
        stamp = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        if self._last_stamp is not None and stamp <= self._last_stamp:
            return
        if self.R_base_cam is None and not self._lookup_extrinsics():
            return
        pose = self.odom.interpolate(stamp, self.max_extra)
        if pose is None:
            if self.odom.empty or stamp < self.odom.oldest:
                self._last_stamp = stamp  # za stare zdjecie -- pomijamy
            return  # odometria jeszcze nie dogonila zdjecia -- sprobujemy w nastepnym takcie
        self._last_stamp = stamp
        self._process(msg, pose)

    def _global_from_odom(self, stamp_msg):
        """T^global_odom (z AMCL) na chwile zdjecia; tozsamosc, gdy global_frame == odom."""
        if self.global_frame in ('', self.odom_frame):
            return np.eye(3), np.zeros(3), self.odom_frame
        tf = None
        for when in (Time.from_msg(stamp_msg), Time()):  # na stemplu, a w razie czego najnowsza
            try:
                tf = self.tf_buffer.lookup_transform(self.global_frame, self.odom_frame, when)
                break
            except Exception:  # noqa: BLE001
                continue
        if tf is None:
            self._tf_fail += 1
            if self._tf_fail % 25 == 1:
                self.get_logger().warn(f'Brak TF {self.global_frame}<-{self.odom_frame} '
                                       '(AMCL?) - pomijam klatke')
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        return rot_from_quat(q.x, q.y, q.z, q.w), np.array([t.x, t.y, t.z]), self.global_frame

    def _process(self, msg, pose):
        g2o = self._global_from_odom(msg.header.stamp)
        if g2o is None:
            return
        R_go, t_go, frame = g2o
        x, y, yaw = pose
        R_gb = R_go @ rot_z(yaw)                      # global <- base_footprint
        t_gb = R_go @ np.array([x, y, 0.0]) + t_go
        R_oc = R_gb @ self.R_base_cam                 # global <- camera_rgb_frame
        t_oc = R_gb @ self.t_base_cam + t_gb
        R_oo = R_oc @ R_CAM_OPT                       # global <- ramka optyczna

        gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        markers = self.detector.detect(gray)

        out = CubeDetectionArray()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = frame
        out.camera_pose.position = Point(x=float(t_oc[0]), y=float(t_oc[1]), z=float(t_oc[2]))
        qx, qy, qz, qw = quat_from_rot(R_oc)
        out.camera_pose.orientation.x, out.camera_pose.orientation.y = float(qx), float(qy)
        out.camera_pose.orientation.z, out.camera_pose.orientation.w = float(qz), float(qw)

        drawn = []
        for marker_id, corners in markers:
            est = self.detector.estimate_pose(corners, self.K, self.D)
            if est is None:
                continue
            R_om, t_om, _err = est
            center, _p, _n = cube_center_from_marker(R_oo, t_oc, R_om, t_om, self.half)
            dist = float(np.linalg.norm(t_om))
            out.detections.append(CubeDetection(
                id=marker_id,
                position=Point(x=float(center[0]), y=float(center[1]), z=float(center[2])),
                distance=dist))
            self._det_count[marker_id] += 1
            self._max_dist[marker_id] = max(self._max_dist[marker_id], dist)
            drawn.append((marker_id, corners, dist))
        self.pub.publish(out)
        self._frames += 1

        if self.dbg_pub.get_subscription_count() > 0:
            self._publish_debug(msg, drawn)

    def _publish_debug(self, msg, drawn):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        for marker_id, corners, dist in drawn:
            pts = corners.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, (0, 255, 0), 3)
            org = tuple(int(v) for v in corners[0])
            cv2.putText(img, f'id {marker_id}  {dist:.2f} m', (org[0], max(30, org[1] - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        if self.debug_scale != 1.0:
            img = cv2.resize(img, None, fx=self.debug_scale, fy=self.debug_scale)
        out = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        out.header = msg.header
        self.dbg_pub.publish(out)

    def _log_stats(self):
        stats = ', '.join(f'{i}: {self._det_count[i]}x (max {self._max_dist[i]:.2f} m)'
                          for i in sorted(self._det_count))
        self.get_logger().info(f'przetworzone klatki: {self._frames}; detekcje: {stats or "-"}')


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
