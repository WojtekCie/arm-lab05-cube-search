import math

import numpy as np

from cube_search.geometry import (R_CAM_OPT, OdomBuffer, cube_center_from_marker, quat_from_rot,
                                  rot_from_quat, rot_z, yaw_from_quat)


def test_quat_roundtrip():
    for yaw in (-3.0, -1.0, 0.0, 0.7, 3.1):
        R = rot_z(yaw) @ R_CAM_OPT
        q = quat_from_rot(R)
        assert np.allclose(rot_from_quat(*q), R, atol=1e-9)
        qz = quat_from_rot(rot_z(yaw))
        assert math.isclose(yaw_from_quat(*qz), yaw, abs_tol=1e-9)


def test_optical_axes():
    # os optyczna z (przod) = x camera_rgb_frame, x optyczna (prawo) = -y, y optyczna (dol) = -z
    assert np.allclose(R_CAM_OPT @ [0, 0, 1], [1, 0, 0])
    assert np.allclose(R_CAM_OPT @ [1, 0, 0], [0, -1, 0])
    assert np.allclose(R_CAM_OPT @ [0, 1, 0], [0, 0, -1])


def test_odom_interpolation_with_wrap():
    buf = OdomBuffer(max_age=2.0)
    buf.add(10.0, 0.0, 0.0, 3.0)
    buf.add(10.1, 1.0, 2.0, -3.0)          # przejscie przez +-pi
    x, y, yaw = buf.interpolate(10.05)
    assert math.isclose(x, 0.5) and math.isclose(y, 1.0)
    assert abs(abs(yaw) - math.pi) < 1e-6  # srodek luku przez pi, a nie przez 0
    assert buf.interpolate(9.0) is None
    assert buf.interpolate(10.12) is not None   # mala ekstrapolacja dozwolona
    assert buf.interpolate(10.3) is None


def test_cube_center_side_face():
    # kamera w (1, 2, 0.117), patrzy wzdluz +y swiata (yaw = 90 st.)
    R_w_cam = rot_z(math.pi / 2)
    t_w_cam = np.array([1.0, 2.0, 0.117])
    R_w_opt = R_w_cam @ R_CAM_OPT
    # marker 2 m przed kamera, zwrocony do niej (z markera = -z optycznej)
    R_opt_m = np.diag([1.0, -1.0, -1.0])
    t_opt_m = np.array([0.0, 0.0, 2.0])
    center, marker, normal = cube_center_from_marker(R_w_opt, t_w_cam, R_opt_m, t_opt_m, 0.045)
    assert np.allclose(marker, [1.0, 4.0, 0.117])
    assert np.allclose(normal, [0.0, -1.0, 0.0])
    assert np.allclose(center, [1.0, 4.045, 0.117])
