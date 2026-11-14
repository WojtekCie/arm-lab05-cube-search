"""Fuzja wielu obserwacji kostek w jedna estymate na ID."""
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class CubeEstimate:
    id: int
    x: float
    y: float
    z: float
    n: int
    std: float
    min_distance: float


class CubeTracker:
    """Bramka wysokosci (kostka lezy na podlodze, srodek na ~0.045 m), mediana per ID,
    odrzucenie obserwacji dalej niz inlier_radius od mediany, srednia wazona 1/d^2
    (blad glebi PnP rosnie z kwadratem odleglosci). Kostka jest potwierdzona po
    min_observations zgodnych obserwacjach, co odsiewa pojedyncze falszywe detekcje."""

    def __init__(self, center_height=0.045, z_gate=0.06, max_distance=4.5, inlier_radius=0.25,
                 min_observations=3, max_observations=400):
        self.center_height = float(center_height)
        self.z_gate = float(z_gate)
        self.max_distance = float(max_distance)
        self.inlier_radius = float(inlier_radius)
        self.min_observations = int(min_observations)
        self.max_observations = int(max_observations)
        self.reset()

    def reset(self):
        self._obs = defaultdict(list)

    def add(self, cube_id, x, y, z, distance):
        values = np.array([x, y, z, distance], dtype=float)
        if cube_id < 0 or not np.all(np.isfinite(values)):
            return False
        if abs(z - self.center_height) > self.z_gate:
            return False
        if not 0.0 < distance <= self.max_distance:
            return False
        obs = self._obs[int(cube_id)]
        obs.append((float(x), float(y), float(z), float(distance)))
        if len(obs) > self.max_observations:  # zachowaj najblizsze (najdokladniejsze)
            obs.sort(key=lambda o: o[3])
            del obs[self.max_observations:]
        return True

    def estimate(self, cube_id):
        obs = self._obs.get(cube_id)
        if not obs:
            return None
        a = np.asarray(obs)
        med = np.median(a[:, :2], axis=0)
        inl = a[np.linalg.norm(a[:, :2] - med, axis=1) < self.inlier_radius]
        if len(inl) == 0:
            return None
        w = 1.0 / np.maximum(inl[:, 3], 0.3) ** 2
        xy = (inl[:, :2] * w[:, None]).sum(axis=0) / w.sum()
        z = float((inl[:, 2] * w).sum() / w.sum())
        std = float(np.sqrt((w * np.sum((inl[:, :2] - xy) ** 2, axis=1)).sum() / w.sum()))
        return CubeEstimate(int(cube_id), float(xy[0]), float(xy[1]), z, int(len(inl)), std,
                            float(inl[:, 3].min()))

    def estimates(self):
        out = [self.estimate(i) for i in sorted(self._obs)]
        return [e for e in out if e is not None]

    def confirmed(self):
        return [e for e in self.estimates() if e.n >= self.min_observations]
