import numpy as np

from cube_search.coverage import FREE, OCCUPIED, UNKNOWN, CoverageGrid


def make_grid():
    # 6 x 5 m, 0.05 m; sciana w x = 3.0 m (kolumna 60) na calej wysokosci
    w, h = 120, 100
    data = np.zeros((h, w), dtype=np.int8)
    data[:, 60] = 100
    data[0, :] = data[-1, :] = 100
    data[:, 0] = data[:, -1] = 100
    cov = CoverageGrid(fov_deg=45.0, range_max=3.5, target_clearance=0.3)
    cov.set_map(data.ravel(), w, h, 0.05, 0.0, 0.0)
    return cov


def seen(cov, x, y):
    ix, iy = cov.world_to_cell(x, y)
    return bool(cov.seen[iy, ix])


def test_fov_range_and_occlusion():
    cov = make_grid()
    cov.reset(1.0, 2.5)
    new = cov.update(1.0, 2.5, 0.0)
    assert new > 0
    assert seen(cov, 2.5, 2.5)          # przed kamera
    assert not seen(cov, 3.5, 2.5)      # za sciana
    assert not seen(cov, 0.5, 2.5)      # za kamera
    assert not seen(cov, 2.0, 4.5)      # poza stozkiem 45 st.
    # cel ograniczony do skladowej spojnej robota (lewa strona sciany)
    assert cov.target[50, 30] and not cov.target[50, 90]
    assert 0.0 < cov.coverage() < 1.0


def test_occupancy_export():
    cov = make_grid()
    cov.reset(1.0, 2.5)
    cov.update(1.0, 2.5, 0.0)
    occ = cov.to_occupancy()
    ix, iy = cov.world_to_cell(2.5, 2.5)
    assert occ[iy, ix] == FREE
    ix, iy = cov.world_to_cell(4.0, 2.5)
    assert occ[iy, ix] == UNKNOWN
    assert occ[50, 60] == OCCUPIED


def test_full_rotation_increases_coverage():
    cov = make_grid()
    cov.reset(1.5, 2.5)
    for k in range(16):
        cov.update(1.5, 2.5, k * 2 * np.pi / 16)
    assert cov.coverage() > 0.9   # lewe pomieszczenie 3 x 5 m w zasiegu 3.5 m


def test_known_free_mask_excludes_grey_unknown(tmp_path):
    import cv2

    from cube_search.coverage import load_known_free_mask
    img = np.full((4, 6), 254, np.uint8)   # wiersz 0 obrazka = gora mapy
    img[0, :] = 205                         # nieznane (map_server publikuje jako wolne)
    img[3, 0] = 0                           # przeszkoda w lewym dolnym rogu
    cv2.imwrite(str(tmp_path / 'm.pgm'), img)
    (tmp_path / 'm.yaml').write_text('image: m.pgm\nresolution: 0.05\norigin: [0, 0, 0]\n'
                                     'negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n')
    mask = load_known_free_mask(str(tmp_path / 'm.yaml'), 6, 4)
    assert not mask[3].any()        # gorny wiersz obrazka = ostatni wiersz siatki
    assert not mask[0, 0]           # przeszkoda trafia do wiersza 0 siatki
    assert mask[0, 1] and mask[1:3].all()
    assert load_known_free_mask(str(tmp_path / 'm.yaml'), 7, 4) is None
