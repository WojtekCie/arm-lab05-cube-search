"""Narzedzia geometryczne (bez zaleznosci od ROS)."""
import bisect
import math

import numpy as np

# Kolumny = osie ramki optycznej (x-prawo, y-dol, z-przod) wyrazone w camera_rgb_frame
# (x-przod, y-lewo, z-gora). Obraz z Gazebo ma frame_id=camera_rgb_frame, a OpenCV zwraca
# poze markera w konwencji optycznej, wiec ten obrot trzeba dolozyc jawnie.
R_CAM_OPT = np.array([[0.0, 0.0, 1.0],
                      [-1.0, 0.0, 0.0],
                      [0.0, -1.0, 0.0]])


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def yaw_from_quat(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_from_yaw(yaw):
    return (0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))


def rot_z(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_from_quat(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_from_rot(R):
    """Macierz obrotu -> kwaternion (x, y, z, w) (metoda Shepperda)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        return (0.25 * s, (R[0, 1] + R[1, 0]) / s,
                (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s)
    if R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        return ((R[0, 1] + R[1, 0]) / s, 0.25 * s,
                (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s)
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
    return ((R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s,
            0.25 * s, (R[1, 0] - R[0, 1]) / s)


class OdomBuffer:
    """Krotka historia poz 2D (x, y, yaw) z /odom z interpolacja na zadanym stemplu.

    Obraz i odometria przychodza z roznymi stemplami; przy obrocie 0.6 rad/s rozjazd
    100 ms daje ~3.4 deg bledu kierunku, czyli ~18 cm na 3 m. Dlatego poze robota
    interpolujemy dokladnie na chwile wykonania zdjecia.
    """

    def __init__(self, max_age=3.0):
        self.max_age = float(max_age)
        self._t = []
        self._p = []

    def clear(self):
        self._t.clear()
        self._p.clear()

    @property
    def empty(self):
        return not self._t

    @property
    def oldest(self):
        return self._t[0] if self._t else None

    @property
    def newest(self):
        return self._t[-1] if self._t else None

    def add(self, t, x, y, yaw):
        if self._t and t <= self._t[-1]:
            if t < self._t[-1] - 1.0:  # zegar cofnal sie (restart symulacji)
                self.clear()
            else:
                return
        self._t.append(float(t))
        self._p.append((float(x), float(y), float(yaw)))
        cut = bisect.bisect_left(self._t, t - self.max_age)
        if cut > 0:
            del self._t[:cut]
            del self._p[:cut]

    def interpolate(self, t, max_extrapolation=0.05):
        """Zwraca (x, y, yaw) w chwili t albo None, gdy t lezy poza buforem."""
        if not self._t or t < self._t[0] or t > self._t[-1] + max_extrapolation:
            return None
        if t >= self._t[-1]:
            return self._p[-1]
        i = bisect.bisect_right(self._t, t)
        t0, t1 = self._t[i - 1], self._t[i]
        (x0, y0, a0), (x1, y1, a1) = self._p[i - 1], self._p[i]
        u = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        return (x0 + u * (x1 - x0), y0 + u * (y1 - y0),
                wrap_angle(a0 + u * wrap_angle(a1 - a0)))


def cube_center_from_marker(R_w_opt, t_w_opt, R_opt_m, t_opt_m, half_size=0.045,
                            top_threshold=0.7):
    """Srodek kostki w ramce swiata na podstawie pozy markera z solvePnP.

    R_w_opt, t_w_opt -- poza ramki optycznej kamery w ramce swiata,
    R_opt_m, t_opt_m -- poza markera w ramce optycznej (os z markera = normalna
                        sciany skierowana na zewnatrz kostki).
    Kostka stoi na podlodze, wiec dla scian bocznych normalna rzutujemy na poziom
    (usuwa to blad nachylenia z niejednoznacznosci PnP); dla sciany gornej srodek
    lezy pionowo pod markerem. Zwraca (srodek, polozenie_markera, normalna).
    """
    t_w_opt = np.asarray(t_w_opt, dtype=float).reshape(3)
    p = R_w_opt @ np.asarray(t_opt_m, dtype=float).reshape(3) + t_w_opt
    n = R_w_opt @ np.asarray(R_opt_m, dtype=float)[:, 2]
    to_cam = t_w_opt - p
    if np.dot(n, to_cam) < 0.0:  # widoczna sciana musi byc zwrocona do kamery
        n = -n
    if n[2] > top_threshold:
        center = p - np.array([0.0, 0.0, half_size])
    else:
        nh = np.array([n[0], n[1], 0.0])
        if np.linalg.norm(nh) < 1e-6:
            nh = np.array([to_cam[0], to_cam[1], 0.0])
        nh /= np.linalg.norm(nh)
        center = p - half_size * nh
    return center, p, n
