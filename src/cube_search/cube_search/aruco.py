"""Detekcja markerow ArUco i estymacja ich pozy (OpenCV 4.6 i nowsze)."""
import cv2
import numpy as np

# Czarny kwadrat markera zajmuje 588/755 px tekstury sciany 9 cm -> bok 0.0701 m.
DEFAULT_MARKER_LENGTH = 0.09 * 588.0 / 755.0


class ArucoDetector:
    def __init__(self, marker_length=DEFAULT_MARKER_LENGTH, dictionary='DICT_ARUCO_ORIGINAL',
                 min_marker_px=12.0, max_reproj_err=3.0):
        aruco = cv2.aruco
        dict_id = getattr(aruco, dictionary)
        if hasattr(aruco, 'getPredefinedDictionary'):
            self.dictionary = aruco.getPredefinedDictionary(dict_id)
        else:  # bardzo stare API
            self.dictionary = aruco.Dictionary_get(dict_id)
        if hasattr(aruco, 'DetectorParameters_create'):  # OpenCV < 4.7
            self.params = aruco.DetectorParameters_create()
        else:
            self.params = aruco.DetectorParameters()
        self.params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        self._detector = (aruco.ArucoDetector(self.dictionary, self.params)
                          if hasattr(aruco, 'ArucoDetector') else None)
        self.marker_length = float(marker_length)
        self.min_marker_px = float(min_marker_px)
        self.max_reproj_err = float(max_reproj_err)
        h = self.marker_length / 2.0
        # Kolejnosc wymagana przez SOLVEPNP_IPPE_SQUARE == kolejnosc naroznikow ArUco.
        self.object_points = np.array([[-h, h, 0.0], [h, h, 0.0],
                                       [h, -h, 0.0], [-h, -h, 0.0]])

    def detect(self, gray):
        """Zwraca liste (id, narozniki 4x2) dla markerow o boku >= min_marker_px."""
        if self._detector is not None:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary,
                                                      parameters=self.params)
        out = []
        if ids is None:
            return out
        for c, marker_id in zip(corners, ids.ravel()):
            c = np.asarray(c, dtype=np.float64).reshape(4, 2)
            side = float(np.mean(np.linalg.norm(c - np.roll(c, 1, axis=0), axis=1)))
            if side >= self.min_marker_px:
                out.append((int(marker_id), c))
        return out

    def estimate_pose(self, corners, K, D):
        """Poza markera w ramce optycznej: (R 3x3, t (3,), blad reprojekcji [px]) albo None."""
        ok, rvec, tvec = cv2.solvePnP(self.object_points, corners, K, D,
                                      flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok or tvec[2, 0] <= 0.0:
            return None
        proj, _ = cv2.projectPoints(self.object_points, rvec, tvec, K, D)
        err = float(np.max(np.linalg.norm(proj.reshape(4, 2) - corners, axis=1)))
        if err > self.max_reproj_err:
            return None
        R, _ = cv2.Rodrigues(rvec)
        return R, tvec.reshape(3), err
