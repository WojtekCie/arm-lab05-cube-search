"""Node cube_search_server: akcja /search_cubes (cube_search_interfaces/action/SearchCubes).

Przebieg akcji:
  1. reset mapy pokrycia i estymat kostek,
  2. SCAN: obrot 360 st. w miejscu (/cmd_vel, kat z /odom),
  3. EXPLORE: publikacja mapy pokrycia kamery (niewidziane = unknown) dla explore_lite,
     ktory wybiera frontiery i wysyla cele do Nav2; po kazdym osiagnietym celu
     explore_lite jest wstrzymywany i robot wykonuje kolejny obrot 360 st.,
  4. koniec: wszystkie oczekiwane kostki potwierdzone / pokrycie >= progu /
     explore_lite nie ma frontierow / stagnacja / limit czasu / anulowanie.
Detekcje (z aruco_detector) sa fuzjonowane przez caly czas trwania akcji.
"""
import array
import math
import os
import threading
import time

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus, GoalStatusArray
from explore_lite_msgs.msg import ExploreStatus
from geometry_msgs.msg import PointStamped, TwistStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from cube_search_interfaces.action import SearchCubes
from cube_search_interfaces.msg import Cube, CubeDetectionArray

from .coverage import CoverageGrid, load_known_free_mask
from .geometry import wrap_angle, yaw_from_quat
from .tracker import CubeTracker

CMD_FRAME = 'cube_search'  # znacznik naszych wiadomosci na /cmd_vel
NAV_ACTIVE = (GoalStatus.STATUS_ACCEPTED, GoalStatus.STATUS_EXECUTING,
              GoalStatus.STATUS_CANCELING)


class CubeSearchServer(Node):
    def __init__(self):
        super().__init__('cube_search_server')
        p = self.declare_parameter
        p('map_topic', '/map')
        p('detections_topic', '/cube_search/detections')
        p('odom_topic', '/odom')
        p('cmd_vel_topic', '/cmd_vel')
        p('coverage_topic', '/cube_search/coverage_map')
        p('markers_topic', '/cube_search/cubes')
        p('explore_resume_topic', '/explore/resume')
        p('explore_status_topic', '/explore/status')
        p('nav_status_topic', '/navigate_to_pose/_action/status')
        p('fov_deg', 45.0)
        p('range_max', 3.5)
        p('ray_step_deg', 0.5)
        p('target_clearance', 0.30)
        p('coverage_publish_period', 1.0)
        p('scan_speed', 0.6)
        p('scan_on_goal_reached', True)
        p('scan_min_travel', 1.0)
        p('stall_timeout', 120.0)
        p('stall_min_gain', 0.005)
        p('min_observations', 3)
        p('inlier_radius', 0.25)
        p('z_gate', 0.06)
        p('max_detection_distance', 4.5)
        p('run_log_dir', os.path.expanduser('~/.ros/cube_search'))
        p('map_yaml', '')  # pusty -> share/arm05_sim/map/map.yaml, jesli istnieje
        g = lambda name: self.get_parameter(name).value  # noqa: E731
        self.g = g

        self.coverage = CoverageGrid(fov_deg=g('fov_deg'), range_max=g('range_max'),
                                     ray_step_deg=g('ray_step_deg'),
                                     target_clearance=g('target_clearance'))
        self.tracker = CubeTracker(z_gate=g('z_gate'), max_distance=g('max_detection_distance'),
                                   inlier_radius=g('inlier_radius'),
                                   min_observations=g('min_observations'))
        self.lock = threading.RLock()
        self.map_msg = None
        self.cov_frame = 'map'
        self.det_frame = 'odom'
        self.odom_pose = None          # (x, y, yaw) w ramce odom
        self.odom_frame = 'odom'
        self.last_det_wall = None
        self.last_foreign_cmd = 0.0
        self.active = False
        self.exploring = False
        self.scanning = False
        self.explore_complete = False
        self.nav_goal_reached = False
        self.nav_busy = False
        self.nav_goals = {}
        self.state = 'IDLE'
        self.trajectory = []
        self._tf_warned = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        cbg = ReentrantCallbackGroup()
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        status_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                                durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, g('map_topic'), self._on_map, latched,
                                 callback_group=cbg)
        self.create_subscription(CubeDetectionArray, g('detections_topic'), self._on_detections,
                                 20, callback_group=cbg)
        self.create_subscription(Odometry, g('odom_topic'), self._on_odom, 20, callback_group=cbg)
        self.create_subscription(TwistStamped, g('cmd_vel_topic'), self._on_cmd_vel, 20,
                                 callback_group=cbg)
        self.create_subscription(ExploreStatus, g('explore_status_topic'),
                                 self._on_explore_status, status_qos, callback_group=cbg)
        self.create_subscription(GoalStatusArray, g('nav_status_topic'), self._on_nav_status,
                                 latched, callback_group=cbg)
        self.cmd_pub = self.create_publisher(TwistStamped, g('cmd_vel_topic'), 10)
        self.cov_pub = self.create_publisher(OccupancyGrid, g('coverage_topic'), latched)
        self.resume_pub = self.create_publisher(Bool, g('explore_resume_topic'), 10)
        self.marker_pub = self.create_publisher(MarkerArray, g('markers_topic'), 10)
        self.create_timer(float(g('coverage_publish_period')), self.publish_coverage,
                          callback_group=cbg)
        self.create_timer(1.0, self._publish_markers, callback_group=cbg)

        self.action_server = ActionServer(
            self, SearchCubes, 'search_cubes', execute_callback=self._execute,
            goal_callback=self._on_goal, cancel_callback=lambda _req: CancelResponse.ACCEPT,
            callback_group=cbg)
        self.get_logger().info('Serwer akcji /search_cubes gotowy.')

    # ============================================================ subskrypcje
    def _on_map(self, msg):
        with self.lock:
            if self.active:
                return
            self.map_msg = msg
            self.cov_frame = msg.header.frame_id or 'map'
            info = msg.info
            known_free = self._known_free_mask(info.width, info.height)
            self.coverage.set_map(msg.data, info.width, info.height, info.resolution,
                                  info.origin.position.x, info.origin.position.y, known_free)
            self.get_logger().info(
                f'Mapa {info.width}x{info.height} @ {info.resolution:.2f} m; pole docelowe '
                f'{np.count_nonzero(self.coverage.target_all) * info.resolution ** 2:.0f} m^2')

    def _known_free_mask(self, width, height):
        path = self.g('map_yaml')
        if not path:
            try:
                from ament_index_python.packages import get_package_share_directory
                path = os.path.join(get_package_share_directory('arm05_sim'), 'map', 'map.yaml')
            except Exception:  # noqa: BLE001
                return None
        try:
            mask = load_known_free_mask(path, width, height)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'Nie wczytano maski wolnej przestrzeni z {path}: {exc!r}')
            return None
        if mask is None:
            self.get_logger().warn(f'Mapa {path} nie pasuje do /map - uzywam samego /map')
        else:
            self.get_logger().info(f'Maska wolnej przestrzeni z {path} '
                                   f'({mask.sum() * 0.0025:.0f} m^2 przy 5 cm)')
        return mask

    def _on_odom(self, msg):
        q = msg.pose.pose.orientation
        self.odom_frame = msg.header.frame_id or 'odom'
        self.odom_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y,
                          yaw_from_quat(q.x, q.y, q.z, q.w))

    def _on_cmd_vel(self, msg):
        if msg.header.frame_id != CMD_FRAME:
            self.last_foreign_cmd = time.monotonic()

    def _on_explore_status(self, msg):
        self.get_logger().info(f'explore_lite: {msg.status}')
        if (msg.status == ExploreStatus.EXPLORATION_COMPLETE and self.active
                and self.exploring and not self.scanning):
            self.explore_complete = True

    def _on_nav_status(self, msg):
        busy = False
        reached = False
        for st in msg.status_list:
            key = bytes(bytearray(st.goal_info.goal_id.uuid))
            if st.status in NAV_ACTIVE:
                busy = True
            if self.nav_goals.get(key) != st.status:
                self.nav_goals[key] = st.status
                if (st.status == GoalStatus.STATUS_SUCCEEDED and self.exploring
                        and not self.scanning):
                    reached = True
        self.nav_busy = busy
        if reached:
            self.nav_goal_reached = True

    def _on_detections(self, msg):
        self.last_det_wall = time.monotonic()
        if not self.active:
            return
        self.det_frame = msg.header.frame_id or 'odom'
        c = msg.camera_pose
        q = c.orientation
        pose = self._to_cov_frame(c.position.x, c.position.y, yaw_from_quat(q.x, q.y, q.z, q.w),
                                  self.det_frame)
        with self.lock:
            if pose is not None:
                self.coverage.update(*pose)
                if (not self.trajectory or math.hypot(pose[0] - self.trajectory[-1][0],
                                                      pose[1] - self.trajectory[-1][1]) > 0.1):
                    self.trajectory.append(pose[:2])
            for d in msg.detections:
                self.tracker.add(d.id, d.position.x, d.position.y, d.position.z, d.distance)

    # ============================================================== pomocnicze
    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _to_cov_frame(self, x, y, yaw, src_frame):
        """Poza 2D z ramki src_frame (odom) do ramki mapy pokrycia (map, przez TF z AMCL)."""
        if src_frame == self.cov_frame:
            return x, y, yaw
        try:
            tf = self.tf_buffer.lookup_transform(self.cov_frame, src_frame, Time())
        except Exception:  # noqa: BLE001
            if not self._tf_warned:
                self.get_logger().warn(f'Brak TF {self.cov_frame}<-{src_frame}; przyjmuje '
                                       'tozsamosc (odom == map)')
                self._tf_warned = True
            return x, y, yaw
        t, q = tf.transform.translation, tf.transform.rotation
        a = yaw_from_quat(q.x, q.y, q.z, q.w)
        ca, sa = math.cos(a), math.sin(a)
        return t.x + ca * x - sa * y, t.y + sa * x + ca * y, wrap_angle(yaw + a)

    def _coverage_value(self):
        with self.lock:
            return self.coverage.coverage()

    def _wait_until(self, pred, timeout, goal_handle=None):
        t0 = time.monotonic()
        while rclpy.ok() and time.monotonic() - t0 < timeout:
            if pred():
                return True
            if goal_handle is not None and goal_handle.is_cancel_requested:
                return False
            time.sleep(0.1)
        return pred()

    def _set_explore(self, enabled):
        self.resume_pub.publish(Bool(data=bool(enabled)))
        self.get_logger().info(f'explore_lite -> {"wznowienie" if enabled else "pauza"}')

    def _publish_cmd(self, wz):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = CMD_FRAME
        msg.twist.angular.z = float(wz)
        self.cmd_pub.publish(msg)

    def _stop_robot(self):
        for _ in range(5):
            self._publish_cmd(0.0)
            time.sleep(0.05)

    def publish_coverage(self):
        if not self.exploring or self.map_msg is None:
            return
        with self.lock:
            occ = self.coverage.to_occupancy()
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.cov_frame
        msg.info = self.map_msg.info
        msg.data = array.array('b', occ.tobytes())
        self.cov_pub.publish(msg)

    def _publish_markers(self):
        with self.lock:
            estimates = self.tracker.estimates()
        min_obs = self.tracker.min_observations
        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        stamp = self.get_clock().now().to_msg()
        for e in estimates:
            ok = e.n >= min_obs
            for k, mtype in enumerate((Marker.CUBE, Marker.TEXT_VIEW_FACING)):
                m = Marker()
                m.header.frame_id = self.det_frame
                m.header.stamp = stamp
                m.ns = 'cubes'
                m.id = 2 * e.id + k
                m.type = mtype
                m.pose.position.x, m.pose.position.y = e.x, e.y
                m.pose.orientation.w = 1.0
                if mtype == Marker.CUBE:
                    m.pose.position.z = 0.045
                    m.scale.x = m.scale.y = m.scale.z = 0.09
                    m.color.r, m.color.g, m.color.b = (0.1, 0.9, 0.1) if ok else (1.0, 0.8, 0.0)
                else:
                    m.pose.position.z = 0.3
                    m.scale.z = 0.2
                    m.color.r = m.color.g = m.color.b = 1.0
                    m.text = f'ID {e.id} ({e.n})'
                m.color.a = 1.0
                arr.markers.append(m)
        self.marker_pub.publish(arr)

    # ================================================================ akcja
    def _on_goal(self, _req):
        if self.active:
            self.get_logger().warn('Odrzucam cel: poprzednie wyszukiwanie wciaz trwa.')
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _scan(self, goal_handle, angle=2.0 * math.pi):
        """Obrot w miejscu o zadany kat; kat liczony z /odom."""
        self.scanning = True
        try:
            t0 = time.monotonic()  # najpierw niech Nav2/collision_monitor przestana publikowac
            while time.monotonic() - self.last_foreign_cmd < 0.6 and time.monotonic() - t0 < 8.0:
                time.sleep(0.05)
            speed = float(self.g('scan_speed'))
            prev = self.odom_pose[2]
            turned = 0.0
            deadline = self._now() + angle / speed * 2.5 + 10.0
            while turned < angle and rclpy.ok() and not goal_handle.is_cancel_requested:
                if self._now() > deadline:
                    self.get_logger().warn(f'Obrot przerwany po {math.degrees(turned):.0f} st.')
                    break
                self._publish_cmd(min(speed, max(0.15, 2.0 * (angle - turned))))
                time.sleep(0.05)
                yaw = self.odom_pose[2]
                turned += abs(wrap_angle(yaw - prev))
                prev = yaw
        finally:
            self._stop_robot()
            self.scanning = False

    def _feedback(self, goal_handle, cov, confirmed):
        fb = SearchCubes.Feedback()
        fb.state = self.state
        fb.coverage = float(cov)
        fb.found_ids = [int(e.id) for e in confirmed]
        goal_handle.publish_feedback(fb)

    def _run(self, goal_handle, expected, cov_goal, timeout, t_start):
        """Wlasciwe przeszukiwanie. Zwraca (status, komunikat)."""
        if not self._wait_until(lambda: self.coverage.ready and self.odom_pose is not None, 30.0,
                                goal_handle):
            return 'aborted', 'Brak mapy (/map) lub odometrii (/odom).'
        if not self._wait_until(lambda: self.last_det_wall is not None
                                and time.monotonic() - self.last_det_wall < 5.0, 30.0,
                                goal_handle):
            return 'aborted', 'Brak danych z aruco_detector (/cube_search/detections).'
        seed = self._to_cov_frame(*self.odom_pose, self.odom_frame)
        with self.lock:
            self.coverage.reset(seed[0], seed[1])
            self.tracker.reset()
            self.trajectory = []
            self.explore_complete = False
            self.nav_goal_reached = False
            self.active = True
        self._set_explore(False)

        self.state = 'SCAN'
        self._scan(goal_handle)
        if goal_handle.is_cancel_requested:
            return 'canceled', 'Anulowano.'

        self.state = 'EXPLORE'
        self.exploring = True
        self.publish_coverage()      # pierwsza mapa uruchamia explore_lite
        time.sleep(0.5)
        self._set_explore(True)
        last_scan_xy = self.odom_pose[:2]
        stall_t, stall_cov = self._now(), self._coverage_value()
        last_fb = last_log = 0.0
        while rclpy.ok():
            time.sleep(0.2)
            if goal_handle.is_cancel_requested:
                return 'canceled', 'Anulowano.'
            now = self._now()
            cov = self._coverage_value()
            with self.lock:
                confirmed = self.tracker.confirmed()
            if time.monotonic() - last_fb > 1.0:
                self._feedback(goal_handle, cov, confirmed)
                last_fb = time.monotonic()
            if time.monotonic() - last_log > 15.0:
                self.get_logger().info(f'[{self.state}] pokrycie {cov:.1%}, potwierdzone ID: '
                                       f'{[e.id for e in confirmed]}')
                last_log = time.monotonic()
            if timeout > 0.0 and now - t_start > timeout:
                return 'aborted', 'Przekroczono limit czasu.'
            if expected > 0 and len(confirmed) >= expected:
                return 'done', f'Potwierdzono wszystkie {expected} oczekiwane kostki.'
            if cov >= cov_goal:
                return 'done', f'Osiagnieto wymagane pokrycie ({cov:.1%}).'
            if self.explore_complete:
                return 'done', f'explore_lite: brak dalszych frontierow (pokrycie {cov:.1%}).'
            if cov - stall_cov > float(self.g('stall_min_gain')):
                stall_t, stall_cov = now, cov
            elif now - stall_t > float(self.g('stall_timeout')):
                return 'done', f'Brak przyrostu pokrycia - stagnacja (pokrycie {cov:.1%}).'
            if self.nav_goal_reached:
                self.nav_goal_reached = False
                x, y = self.odom_pose[:2]
                travel = math.hypot(x - last_scan_xy[0], y - last_scan_xy[1])
                if self.g('scan_on_goal_reached') and travel >= float(self.g('scan_min_travel')):
                    self.state = 'SCAN'
                    self._set_explore(False)
                    self._wait_until(lambda: not self.nav_busy, 10.0, goal_handle)
                    self._scan(goal_handle)
                    last_scan_xy = self.odom_pose[:2]
                    self.state = 'EXPLORE'
                    if not goal_handle.is_cancel_requested:
                        self._set_explore(True)
        return 'aborted', 'Zamykanie node.'

    def _execute(self, goal_handle):
        req = goal_handle.request
        expected = int(req.expected_cubes)
        cov_goal = float(req.coverage_goal) if req.coverage_goal > 0.0 else 0.95
        timeout = float(req.timeout)
        t_start = self._now()
        self.get_logger().info(f'Nowy cel: expected_cubes={expected}, coverage_goal={cov_goal:.2f}, '
                               f'timeout={timeout:.0f} s')
        status, message = 'aborted', 'Blad wewnetrzny.'
        try:
            status, message = self._run(goal_handle, expected, cov_goal, timeout, t_start)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'Wyjatek podczas wyszukiwania: {exc!r}')
            message = f'Wyjatek: {exc!r}'
        finally:
            if self.active:
                self._set_explore(False)
            self.exploring = False
            self._stop_robot()
            self.active = False
            self.state = 'IDLE'

        with self.lock:
            confirmed = self.tracker.confirmed()
            unconfirmed = [e for e in self.tracker.estimates()
                           if e.n < self.tracker.min_observations]
            cov = self.coverage.coverage() if self.coverage.ready else 0.0
        result = SearchCubes.Result()
        result.coverage = float(cov)
        result.elapsed = float(self._now() - t_start)
        stamp = self.get_clock().now().to_msg()
        for e in confirmed:
            c = Cube()
            c.id = e.id
            c.position = PointStamped()
            c.position.header.frame_id = self.det_frame
            c.position.header.stamp = stamp
            c.position.point.x, c.position.point.y, c.position.point.z = e.x, e.y, e.z
            c.observations = e.n
            c.std_dev = float(e.std)
            result.cubes.append(c)
        if unconfirmed:
            message += f' Niepotwierdzone ID: {[e.id for e in unconfirmed]}.'
        message += f' Znaleziono {len(confirmed)} kostek.'
        result.message = message
        result.success = status == 'done' and (expected <= 0 or len(confirmed) >= expected)
        self.get_logger().info(f'Koniec ({status}): {message} Czas {result.elapsed:.0f} s.')
        self._save_run(result)
        if status == 'canceled':
            goal_handle.canceled()
        elif status == 'done':
            goal_handle.succeed()
        else:
            goal_handle.abort()
        return result

    def _save_run(self, result):
        """Zapis przebiegu (pokrycie, trajektoria, kostki) do wizualizacji w raporcie."""
        try:
            out_dir = self.g('run_log_dir')
            os.makedirs(out_dir, exist_ok=True)
            with self.lock:
                cov = self.coverage
                np.savez_compressed(
                    os.path.join(out_dir, 'last_run.npz'), free=cov.free, seen=cov.seen,
                    target=cov.target, res=cov.res, origin=np.array([cov.ox, cov.oy]),
                    trajectory=np.array(self.trajectory, dtype=float).reshape(-1, 2),
                    cubes=np.array([[c.id, c.position.point.x, c.position.point.y]
                                    for c in result.cubes], dtype=float).reshape(-1, 3),
                    coverage=result.coverage, elapsed=result.elapsed)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'Nie zapisano przebiegu: {exc!r}')


def main(args=None):
    rclpy.init(args=args)
    node = CubeSearchServer()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
