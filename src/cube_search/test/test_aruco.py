"""Syntetyczny test detekcji: sciana kostki (tekstura z symulacji albo wygenerowana)
rzutowana kamera o parametrach z /camera/camera_info."""
import math
import os

import cv2
import numpy as np
import pytest

from cube_search.aruco import DEFAULT_MARKER_LENGTH, ArucoDetector
from cube_search.geometry import R_CAM_OPT, cube_center_from_marker

K = np.array([[1696.8028, 0.0, 960.0], [0.0, 1696.8027, 540.0], [0.0, 0.0, 1.0]])
D = np.zeros(5)
FACE = 0.09
TEXTURE = '/arm_ws/src/arm05_sim/models/aruco_markers/{id}/materials/textures/{id}.png'


def face_texture(marker_id):
    path = TEXTURE.format(id=marker_id)
    if os.path.exists(path):
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    gen = getattr(cv2.aruco, 'generateImageMarker', None) or cv2.aruco.drawMarker
    marker = gen(d, marker_id, 588)
    tex = np.full((755, 755), 255, np.uint8)
    tex[84:672, 84:672] = marker
    return tex


def ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def render(marker_id, R_opt_m, t_opt_m):
    tex = face_texture(marker_id)
    a = FACE / 2
    face = np.array([[-a, a, 0], [a, a, 0], [a, -a, 0], [-a, -a, 0]])
    pts_cam = (R_opt_m @ face.T).T + t_opt_m
    uv = (K @ pts_cam.T).T
    uv = (uv[:, :2] / uv[:, 2:]).astype(np.float32)
    w = tex.shape[1]
    src = np.array([[0, 0], [w, 0], [w, w], [0, w]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, uv)
    img = np.full((1080, 1920), 140, np.uint8)
    warped = cv2.warpPerspective(tex, H, (1920, 1080), borderValue=0)
    mask = cv2.warpPerspective(np.full_like(tex, 255), H, (1920, 1080), borderValue=0)
    img[mask > 0] = warped[mask > 0]
    noise = np.random.default_rng(0).normal(0, 2.0, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


@pytest.mark.parametrize('marker_id,angle,dist', [(0, 0.0, 1.0), (3, 30.0, 1.5), (4, -40.0, 2.5)])
def test_detect_and_pose(marker_id, angle, dist):
    R_opt_m = np.diag([1.0, -1.0, -1.0]) @ ry(math.radians(angle))
    t_opt_m = np.array([0.12, 0.05, dist])
    img = render(marker_id, R_opt_m, t_opt_m)
    det = ArucoDetector(DEFAULT_MARKER_LENGTH)
    found = det.detect(img)
    assert [i for i, _ in found] == [marker_id]
    R, t, err = det.estimate_pose(found[0][1], K, D)
    assert err < 1.0
    assert np.linalg.norm(t - t_opt_m) < 0.01 * dist   # < 1 % odleglosci
    # srodek kostki w swiecie (kamera w poczatku ukladu, patrzy wzdluz +x)
    center, _, _ = cube_center_from_marker(R_CAM_OPT, np.zeros(3), R, t, FACE / 2)
    expected = R_CAM_OPT @ (t_opt_m + R_opt_m @ np.array([0.0, 0.0, -FACE / 2]))
    # bledna konwencja osi dalaby >= 4.5 cm (pol boku), zly bok markera ~28 % odleglosci
    assert np.linalg.norm(center - expected) < 0.005 + 0.01 * dist


def test_wrong_marker_length_overestimates_distance():
    """Uzycie 0.09 m (bok sciany) zamiast 0.0701 m zawyza odleglosc o ~28 %."""
    R_opt_m = np.diag([1.0, -1.0, -1.0])
    t_opt_m = np.array([0.0, 0.0, 1.5])
    img = render(1, R_opt_m, t_opt_m)
    det = ArucoDetector(0.09)
    _, t, _ = det.estimate_pose(det.detect(img)[0][1], K, D)
    assert 1.25 < t[2] / 1.5 < 1.31
