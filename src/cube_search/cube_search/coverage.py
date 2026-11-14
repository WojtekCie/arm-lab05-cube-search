"""Mapa pokrycia kamery: ktore komorki wolnej przestrzeni byly juz w polu widzenia."""
import math

import numpy as np
from scipy import ndimage

FREE, OCCUPIED, UNKNOWN = 0, 100, -1


def load_known_free_mask(map_yaml, width, height, max_occupancy=0.1):
    """Maska 'pewnie wolnych' komorek z pliku mapy (yaml + pgm), w ukladzie OccupancyGrid
    (wiersz 0 = dol mapy). Potrzebna, bo map_server z free_thresh=0.25 publikuje szary
    obszar nieznany (205 -> p=0.196) jako WOLNY, a dom ma przerwy w scianach -- bez tej
    maski pole docelowe obejmowaloby teren na zewnatrz budynku."""
    import os

    import cv2
    import yaml

    with open(map_yaml) as f:
        cfg = yaml.safe_load(f)
    image = cfg['image']
    if not os.path.isabs(image):
        image = os.path.join(os.path.dirname(map_yaml), image)
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    if img is None or img.shape != (height, width):
        return None
    shade = img[::-1].astype(np.float64) / 255.0
    occ = shade if cfg.get('negate', 0) else 1.0 - shade
    return occ < max_occupancy


class CoverageGrid:
    """Siatka zgodna z /map. Komorka jest 'widziana', gdy lezala w stozku kamery
    (FOV, zasieg) i promien z kamery nie przecial po drodze przeszkody z mapy.

    Eksport do OccupancyGrid dla explore_lite: widziana wolna = 0, niewidziana
    wolna = -1 (unknown -> frontier), wszystko inne = 100, wiec klasyczne frontier
    exploration staje sie eksploracja pokrycia kamery.
    """

    def __init__(self, fov_deg=45.0, range_max=3.5, ray_step_deg=0.5, target_clearance=0.30,
                 robot_body_radius=0.2, free_threshold=25):
        self.fov = math.radians(fov_deg)
        self.range_max = float(range_max)
        self.ray_step = math.radians(ray_step_deg)
        self.target_clearance = float(target_clearance)
        self.robot_body_radius = float(robot_body_radius)
        self.free_threshold = int(free_threshold)
        self.ready = False

    # ------------------------------------------------------------------ mapa
    def set_map(self, data, width, height, resolution, origin_x, origin_y, known_free=None):
        grid = np.asarray(data, dtype=np.int16).reshape(height, width)
        self.W, self.H = int(width), int(height)
        self.res = float(resolution)
        self.ox, self.oy = float(origin_x), float(origin_y)
        self.free = (grid >= 0) & (grid <= self.free_threshold)
        if known_free is not None:
            self.free &= known_free
        self.blocking = ~self.free  # przeszkody i obszar nieznany zaslaniaja widok
        clearance = ndimage.distance_transform_edt(self.free) * self.res
        # Pole docelowe: wolne komorki z odstepem od przeszkod (tam moze lezec kostka).
        self.target_all = self.free & (clearance >= self.target_clearance)
        self.target = self.target_all.copy()
        self.seen = np.zeros_like(self.free)
        n = max(1, int(round(self.fov / self.ray_step)))
        self._offsets = np.linspace(-self.fov / 2.0, self.fov / 2.0, n + 1)
        self._ranges = np.arange(0.0, self.range_max + 1e-9, 0.5 * self.res)
        self._body = self._ranges < self.robot_body_radius
        self.ready = True

    def world_to_cell(self, x, y):
        ix = int(math.floor((x - self.ox) / self.res))
        iy = int(math.floor((y - self.oy) / self.res))
        if 0 <= ix < self.W and 0 <= iy < self.H:
            return ix, iy
        return None

    def reset(self, x=None, y=None):
        """Czysci pokrycie; opcjonalnie zaweza pole docelowe do skladowej spojnej z (x, y)."""
        self.seen[:] = False
        self.target = self.target_all.copy()
        if x is None or y is None:
            return
        cell = self.world_to_cell(x, y)
        if cell is None:
            return
        labels, _ = ndimage.label(self.free, structure=np.ones((3, 3)))
        ix, iy = cell
        label = labels[iy, ix]
        if label == 0:  # robot "w scianie" (blad lokalizacji) -> najblizsza wolna komorka
            idx = ndimage.distance_transform_edt(~self.free, return_distances=False,
                                                 return_indices=True)
            label = labels[idx[0][iy, ix], idx[1][iy, ix]]
        if label > 0:
            self.target = self.target_all & (labels == label)

    # ------------------------------------------------------------- pokrycie
    def update(self, x, y, yaw):
        """Ray casting stozka kamery z pozycji (x, y, yaw). Zwraca liczbe nowych komorek celu."""
        if not self.ready:
            return 0
        ang = yaw + self._offsets
        xs = x + np.outer(np.cos(ang), self._ranges)
        ys = y + np.outer(np.sin(ang), self._ranges)
        ix = np.floor((xs - self.ox) / self.res).astype(np.int64)
        iy = np.floor((ys - self.oy) / self.res).astype(np.int64)
        inside = (ix >= 0) & (ix < self.W) & (iy >= 0) & (iy < self.H)
        ixc = np.clip(ix, 0, self.W - 1)
        iyc = np.clip(iy, 0, self.H - 1)
        # Przeszkody w obrebie korpusu robota ignorujemy (blad lokalizacji przy scianie).
        block = (self.blocking[iyc, ixc] & ~self._body[None, :]) | ~inside
        visible = ~np.logical_or.accumulate(block, axis=1)
        vy, vx = iyc[visible], ixc[visible]
        new = int(np.count_nonzero(self.target[vy, vx] & ~self.seen[vy, vx]))
        self.seen[vy, vx] = True
        return new

    def coverage(self):
        total = int(np.count_nonzero(self.target))
        if total == 0:
            return 0.0
        return float(np.count_nonzero(self.seen & self.target)) / total

    def to_occupancy(self):
        occ = np.full((self.H, self.W), OCCUPIED, dtype=np.int8)
        occ[self.free & ~self.seen] = UNKNOWN
        occ[self.free & self.seen] = FREE
        return occ
