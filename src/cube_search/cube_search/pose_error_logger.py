"""Node pose_error_logger: ewaluacja lokalizacji i detekcji wzgledem ground truth z Gazebo.

Ground truth pozy robota: topic gz /world/default/dynamic_pose/info (gz.msgs.Pose_V, model
'waffle') czytany bezposrednio przez gz-transport (most ros_gz Pose_V->TFMessage gubi nazwy
modeli i stemple). Swiat Gazebo == map == odom w chwili startu (robot startuje w (0, 0, 0)).

Zapisuje do <out_dir>:
  pose_errors.csv  -- co sample_period [s]: poza GT, poza z samego /odom i z /odom + korekta
                      AMCL (TF map->odom) oraz ich bledy,
  detections.csv   -- kazda detekcja kostki: blad calkowity (tak jak liczy detektor) oraz blad
                      'samej percepcji' (ta sama detekcja przeliczona z prawdziwa poza robota),
                      z rozkladem na skladowa wzdluz promienia widzenia (glebia) i poprzeczna.
"""
import math
import os
import re
from collections import deque

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

from cube_search_interfaces.msg import CubeDetectionArray

from .geometry import OdomBuffer, rot_from_quat, wrap_angle, yaw_from_quat

GT_RE = re.compile(r'Aruco\s+(\d+):\s*x:\s*([-+0-9.eE]+)\s+y:\s*([-+0-9.eE]+)')


class PoseErrorLogger(Node):
    def __init__(self):
        super().__init__('pose_error_logger')
        p = self.declare_parameter
        p('gt_topic', '/world/default/dynamic_pose/info')
        p('gt_model', 'waffle')
        p('odom_topic', '/odom')
        p('detections_topic', '/cube_search/detections')
        p('map_frame', 'map')
        p('odom_frame', 'odom')
        p('sample_period', 0.5)
        p('ground_truth_file', os.path.expanduser('~/.ros/arm05_sim/logs/latest.txt'))
        p('out_dir', os.path.expanduser('~/.ros/cube_search/eval'))
        g = lambda name: self.get_parameter(name).value  # noqa: E731
        self.model = g('gt_model')
        self.map_frame, self.odom_frame = g('map_frame'), g('odom_frame')
        self.period = float(g('sample_period'))
        self.cubes = self._load_cubes(g('ground_truth_file'))

        self.odom = OdomBuffer(max_age=5.0)
        self.gt = OdomBuffer(max_age=5.0)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._last_sample = None
        self._t0 = None
        self.R_bc = None
        self.t_bc = None

        out = g('out_dir')
        os.makedirs(out, exist_ok=True)
        self.f_pose = open(os.path.join(out, 'pose_errors.csv'), 'w')
        self.f_pose.write('t,gt_x,gt_y,gt_yaw,odom_x,odom_y,odom_yaw,amcl_x,amcl_y,amcl_yaw,'
                          'err_odom,err_amcl,yaw_err_odom,yaw_err_amcl\n')
        self.f_det = open(os.path.join(out, 'detections.csv'), 'w')
        self.f_det.write('t,id,distance,err_total,err_percep,err_radial,err_tangential,'
                         'err_robot_amcl\n')

        # gz-transport wola callback z wlasnego watku -> kolejka, oprozniana w timerze ROS
        from gz.msgs10.pose_v_pb2 import Pose_V
        from gz.transport13 import Node as GzNode
        self._gt_queue = deque(maxlen=2000)
        self._gz_node = GzNode()
        if not self._gz_node.subscribe(Pose_V, g('gt_topic'), self._on_gt_gz):
            self.get_logger().error(f'Nie mozna subskrybowac gz {g("gt_topic")}')
        self.create_timer(0.05, self._drain_gt)
        self.create_subscription(Odometry, g('odom_topic'), self._on_odom, 50)
        self.create_subscription(CubeDetectionArray, g('detections_topic'), self._on_det, 20)
        self.create_timer(5.0, self._flush)
        self.get_logger().info(f'Ewaluacja -> {out}; kostki GT: {sorted(self.cubes)}')

    @staticmethod
    def _load_cubes(path):
        cubes = {}
        try:
            with open(path) as f:
                for line in f:
                    m = GT_RE.search(line)
                    if m:
                        cubes[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
        except OSError:
            pass
        return cubes

    # ------------------------------------------------------------ dane
    def _on_odom(self, msg):
        q = msg.pose.pose.orientation
        self.odom.add(Time.from_msg(msg.header.stamp).nanoseconds * 1e-9,
                      msg.pose.pose.position.x, msg.pose.pose.position.y,
                      yaw_from_quat(q.x, q.y, q.z, q.w))

    def _on_gt_gz(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
        for pose in msg.pose:
            if pose.name == self.model:
                q, p = pose.orientation, pose.position
                self._gt_queue.append((t, p.x, p.y, yaw_from_quat(q.x, q.y, q.z, q.w)))
                return

    def _drain_gt(self):
        while self._gt_queue:
            t, x, y, yaw = self._gt_queue.popleft()
            self.gt.add(t, x, y, yaw)
            if (self._last_sample is None or t - self._last_sample >= self.period) \
                    and self._sample(t):
                self._last_sample = t

    def _lookup_base_cam(self):
        try:
            tf = self.tf_buffer.lookup_transform('base_footprint', 'camera_rgb_frame', Time())
        except Exception:  # noqa: BLE001
            return False
        q, p = tf.transform.rotation, tf.transform.translation
        self.R_bc = rot_from_quat(q.x, q.y, q.z, q.w)
        self.t_bc = np.array([p.x, p.y, p.z])
        return True

    def _map_from_odom(self, t):
        for when in (Time(nanoseconds=int(t * 1e9)), Time()):
            try:
                tf = self.tf_buffer.lookup_transform(self.map_frame, self.odom_frame, when)
            except Exception:  # noqa: BLE001
                continue
            q = tf.transform.rotation
            return (tf.transform.translation.x, tf.transform.translation.y,
                    yaw_from_quat(q.x, q.y, q.z, q.w))
        return None

    @staticmethod
    def _compose(a, b):
        ax, ay, ath = a
        bx, by, bth = b
        c, s = math.cos(ath), math.sin(ath)
        return ax + c * bx - s * by, ay + s * bx + c * by, wrap_angle(ath + bth)

    def _poses_at(self, t):
        gt = self.gt.interpolate(t, 0.1)
        od = self.odom.interpolate(t, 0.1)
        mo = self._map_from_odom(t)
        if gt is None or od is None or mo is None:
            return None
        return gt, od, self._compose(mo, od)

    def _sample(self, t):
        poses = self._poses_at(t)
        if poses is None:
            return False
        gt, od, am = poses
        if self._t0 is None:
            self._t0 = t
        e_od = math.hypot(od[0] - gt[0], od[1] - gt[1])
        e_am = math.hypot(am[0] - gt[0], am[1] - gt[1])
        self.f_pose.write(
            f'{t - self._t0:.2f},{gt[0]:.4f},{gt[1]:.4f},{gt[2]:.4f},{od[0]:.4f},{od[1]:.4f},'
            f'{od[2]:.4f},{am[0]:.4f},{am[1]:.4f},{am[2]:.4f},{e_od:.4f},{e_am:.4f},'
            f'{abs(wrap_angle(od[2] - gt[2])):.4f},{abs(wrap_angle(am[2] - gt[2])):.4f}\n')
        return True

    def _on_det(self, msg):
        if not msg.detections or self._t0 is None:
            return
        t = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        poses = self._poses_at(t)
        if poses is None:
            return
        gt, _od, am = poses
        if self.t_bc is None and not self._lookup_base_cam():
            return
        c = msg.camera_pose
        R_used = rot_from_quat(c.orientation.x, c.orientation.y, c.orientation.z,
                               c.orientation.w)
        t_used = np.array([c.position.x, c.position.y, c.position.z])
        # prawdziwa poza kamery = poza robota z Gazebo * stala T_base_cam (tf_static)
        R_gb = np.array([[math.cos(gt[2]), -math.sin(gt[2]), 0.0],
                         [math.sin(gt[2]), math.cos(gt[2]), 0.0], [0.0, 0.0, 1.0]])
        R_gc = R_gb @ self.R_bc
        t_gc = R_gb @ self.t_bc + np.array([gt[0], gt[1], 0.0])
        err_robot = math.hypot(am[0] - gt[0], am[1] - gt[1])
        for d in msg.detections:
            if d.id not in self.cubes:
                continue
            cx, cy = self.cubes[d.id]
            p_used = np.array([d.position.x, d.position.y, d.position.z])
            p_cam = R_used.T @ (p_used - t_used)          # wynik samej percepcji
            p_true_pose = R_gc @ p_cam + t_gc              # ta sama detekcja z prawdziwa poza
            e_tot = math.hypot(p_used[0] - cx, p_used[1] - cy)
            ev = np.array([p_true_pose[0] - cx, p_true_pose[1] - cy])
            ray = np.array([cx - t_gc[0], cy - t_gc[1]])
            ray /= max(np.linalg.norm(ray), 1e-9)
            e_rad = float(np.dot(ev, ray))                 # + = za daleko
            e_tan = float(ray[0] * ev[1] - ray[1] * ev[0])
            self.f_det.write(f'{t - self._t0:.2f},{d.id},{d.distance:.3f},{e_tot:.4f},'
                             f'{float(np.linalg.norm(ev)):.4f},{e_rad:.4f},{e_tan:.4f},'
                             f'{err_robot:.4f}\n')

    def _flush(self):
        self.f_pose.flush()
        self.f_det.flush()

    def destroy_node(self):
        self._flush()
        self.f_pose.close()
        self.f_det.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PoseErrorLogger()
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
