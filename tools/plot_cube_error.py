#!/usr/bin/env python3
"""Blad bezwzgledny pozycji kostki (|p_est - p_GT| w plaszczyznie XY) w funkcji odleglosci
kamera-marker, dla kazdej pojedynczej detekcji z detections.csv (pose_error_logger).

Uzycie:  python3 tools/plot_cube_error.py <katalog z detections.csv> <wyjscie bez rozszerzenia>
Wynik:   <wyjscie>.pdf, <wyjscie>.png (300 dpi) + tabela median w przedzialach na stdout.
"""
import sys

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from plot_eval import C_AMCL, C_PERC, FALSE_ID_ERR, INK, INK2, load, style  # noqa: E402


def binned(x, y, edges, min_n=10):
    xc, q25, med, q75, n = [], [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() >= min_n:
            xc.append(0.5 * (lo + hi))
            a, b, c = np.percentile(y[m], [25, 50, 75])
            q25.append(a), med.append(b), q75.append(c), n.append(int(m.sum()))
    return [np.array(v) for v in (xc, q25, med, q75, n)]


def main():
    src, out = sys.argv[1], sys.argv[2]
    de = load(f'{src}/detections.csv')
    false = de['err_percep'] > FALSE_ID_ERR
    n_false = int(false.sum())
    de = {k: v[~false] for k, v in de.items()}
    d, e_tot, e_per = de['distance'], 100 * de['err_total'], 100 * de['err_percep']

    edges = np.arange(0.0, np.ceil(d.max()) + 0.5, 0.5)
    xc, q25, med, q75, n = binned(d, e_tot, edges)
    _, _, med_p, _, _ = binned(d, e_per, edges)
    print(f'{len(d)} detekcji, pominieto {n_false} falszywych identyfikacji (blad > 1 m)')
    print(' odleglosc [m]    n   Q25   mediana   Q75   (sama percepcja: mediana)  [cm]')
    for i in range(len(xc)):
        print(f'  {xc[i] - 0.25:.2f}-{xc[i] + 0.25:.2f}  {n[i]:4d} {q25[i]:5.1f}  {med[i]:6.1f}  '
              f'{q75[i]:6.1f}   ({med_p[i]:5.1f})')

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5})
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    ax.scatter(d, e_tot, s=7, color=C_AMCL, alpha=0.28, linewidths=0, zorder=2)
    ax.fill_between(xc, q25, q75, color=C_AMCL, alpha=0.18, linewidth=0, zorder=1,
                    label='całkowity: kwartyle 25–75 %')
    ax.plot(xc, med, color=C_AMCL, linewidth=2, marker='o', markersize=3.5, zorder=4,
            label='całkowity (tak jak liczy robot): mediana')
    ax.plot(xc, med_p, color=C_PERC, linewidth=2, marker='o', markersize=3.5, zorder=4,
            label='sama percepcja (prawdziwa poza robota): mediana')
    ax.axvline(4.5, color=INK2, linewidth=0.8, linestyle=(0, (3, 2)), zorder=3)
    ax.text(4.57, 0.97, 'bramka fuzji 4.5 m', transform=ax.get_xaxis_transform(),
            fontsize=6.5, color=INK2, va='top')
    ax.set_xlim(0, np.ceil(d.max()))
    ax.set_ylim(0, None)
    ax.set_xlabel('odległość kamera–marker [m]')
    ax.set_ylabel('błąd bezwzględny pozycji kostki [cm]')
    ax.set_title('Błąd pozycji kostki z pojedynczej detekcji vs odległość',
                 fontsize=8.5, color=INK, loc='left', pad=14)
    ax.text(0.0, 1.015, f'{len(d)} detekcji z jednego przebiegu; pominięto {n_false} '
            f'fałszywych identyfikacji (błąd > 1 m)', transform=ax.transAxes, fontsize=6.5,
            color=INK2, ha='left', va='bottom')
    style(ax)
    h, lab = ax.get_legend_handles_labels()
    order = [1, 0, 2]
    ax.legend([h[i] for i in order], [lab[i] for i in order], frameon=False, fontsize=7,
              labelcolor=INK, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2,
              columnspacing=1.5, handlelength=1.8)
    fig.tight_layout(pad=0.4)
    fig.savefig(f'{out}.pdf')
    fig.savefig(f'{out}.png', dpi=300)
    print(f'zapisano {out}.pdf, {out}.png')


if __name__ == '__main__':
    main()
