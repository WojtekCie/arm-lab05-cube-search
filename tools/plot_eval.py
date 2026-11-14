#!/usr/bin/env python3
"""Wykresy i statystyki z ewaluacji (pose_error_logger): blad lokalizacji w czasie
(sama odometria vs odometria + korekta AMCL) oraz blad detekcji kostek vs odleglosc
(calkowity vs sama percepcja).

Uzycie:  python3 tools/plot_eval.py <katalog z pose_errors.csv i detections.csv> <wyjscie bez rozszerzenia>
Wynik:   <wyjscie>.pdf (do LaTeX), <wyjscie>.png (300 dpi) + statystyki na stdout.
"""
import csv
import sys

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# paleta referencyjna (sloty 1-3, zwalidowane dla wszystkich par), tekst w odcieniach szarosci
C_ODOM, C_AMCL, C_PERC = '#2a78d6', '#eb6834', '#1baf7a'
FALSE_ID_ERR = 1.0  # [m]
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]} if rows else {}


def stats(name, v, unit=100.0, suffix='cm'):
    print(f'{name:34s} sr {unit * v.mean():6.1f}  med {unit * np.median(v):6.1f}  '
          f'RMS {unit * np.sqrt(np.mean(v ** 2)):6.1f}  maks {unit * v.max():6.1f} {suffix}')


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK2)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=7, width=0.6, length=2.5)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def binned_median(x, y, edges, min_n=10):
    xc, med = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() >= min_n:
            xc.append(0.5 * (lo + hi))
            med.append(np.median(y[m]))
    return np.array(xc), np.array(med)


def main():
    src, out = sys.argv[1], sys.argv[2]
    pe = load(f'{src}/pose_errors.csv')
    de = load(f'{src}/detections.csv')

    print(f'czas przebiegu: {pe["t"][-1]:.0f} s symulacji, {len(pe["t"])} probek, '
          f'{len(de.get("t", []))} detekcji')
    stats('blad polozenia: sama odometria', pe['err_odom'])
    stats('blad polozenia: odometria + AMCL', pe['err_amcl'])
    stats('blad kata: sama odometria', pe['yaw_err_odom'], np.degrees(1.0), 'st.')
    stats('blad kata: odometria + AMCL', pe['yaw_err_amcl'], np.degrees(1.0), 'st.')
    n_false = 0
    if de:
        # blad > 1 m = inna kostka/wzor odczytany jako to ID (falszywa identyfikacja)
        false = de['err_percep'] > FALSE_ID_ERR
        n_false = int(false.sum())
        print(f'falszywe identyfikacje (blad > {FALSE_ID_ERR:.0f} m): {n_false} detekcji, ID '
              f'{sorted(set(de["id"][false].astype(int).tolist()))}')
        de = {k: v[~false] for k, v in de.items()}
        stats('detekcja: blad calkowity', de['err_total'])
        stats('detekcja: sama percepcja', de['err_percep'])
        a, b = np.polyfit(de['distance'], de['err_radial'], 1)
        print(f'blad wzdluz promienia ~ {100 * a:+.1f} % odleglosci {100 * b:+.1f} cm '
              f'(ujemny = za blisko); poprzeczny: med |e| '
              f'{100 * np.median(np.abs(de["err_tangential"])):.1f} cm')
        for lo, hi in ((0, 2), (2, 3), (3, 4.5), (4.5, 9)):
            m = (de['distance'] >= lo) & (de['distance'] < hi)
            if m.any():
                print(f'  d {lo:.1f}-{hi:.1f} m: n={m.sum():4d}  percepcja med '
                      f'{100 * np.median(de["err_percep"][m]):5.1f} cm, wzdluz promienia med '
                      f'{100 * np.median(de["err_radial"][m]):+6.1f} cm, calkowity med '
                      f'{100 * np.median(de["err_total"][m]):5.1f} cm')

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 2.6),
                                   gridspec_kw={'width_ratios': [1.35, 1]})
    # (a) blad polozenia robota w czasie
    t = pe['t']
    ax1.plot(t, 100 * pe['err_odom'], color=C_ODOM, linewidth=1.4,
             label=f'sama /odom (śr. {100 * pe["err_odom"].mean():.0f} cm)')
    ax1.plot(t, 100 * pe['err_amcl'], color=C_AMCL, linewidth=1.4,
             label=f'/odom + korekta AMCL (śr. {100 * pe["err_amcl"].mean():.0f} cm)')
    ax1.set_xlabel('czas symulacji [s]')
    ax1.set_ylabel('błąd położenia robota [cm]')
    ax1.set_title('a) Lokalizacja względem pozy z Gazebo', fontsize=8, color=INK, loc='left')
    ax1.set_xlim(0, t[-1])
    ax1.set_ylim(0, None)
    style(ax1)
    ax1.legend(frameon=False, fontsize=7, loc='upper left', labelcolor=INK)
    # (b) blad detekcji vs odleglosc
    if de:
        d = de['distance']
        ax2.scatter(d, 100 * de['err_total'], s=6, color=C_AMCL, alpha=0.25, linewidths=0)
        ax2.scatter(d, 100 * de['err_percep'], s=6, color=C_PERC, alpha=0.25, linewidths=0)
        edges = np.arange(0.0, np.ceil(d.max()) + 0.5, 0.5)
        x, m = binned_median(d, 100 * de['err_total'], edges)
        ax2.plot(x, m, color=C_AMCL, linewidth=1.6, marker='o', markersize=3,
                 label='całkowity (poza z AMCL)')
        x, m = binned_median(d, 100 * de['err_percep'], edges)
        ax2.plot(x, m, color=C_PERC, linewidth=1.6, marker='o', markersize=3,
                 label='sama percepcja (prawdziwa poza)')
        ax2.axvline(4.5, color=INK2, linewidth=0.8, linestyle=(0, (3, 2)))
        ax2.text(4.6, 0.04, 'bramka fuzji 4.5 m', transform=ax2.get_xaxis_transform(),
                 fontsize=6.3, color=INK2, va='bottom',
                 bbox=dict(facecolor='white', edgecolor='none', pad=0.8, alpha=0.85))
        if n_false:
            ax2.text(0.03, 0.78, f'pominięto {n_false} fałszywych identyfikacji (błąd > 1 m)',
                     transform=ax2.transAxes, fontsize=6.3, color=INK2, ha='left', va='top')
        ax2.set_xlim(0, None)
        ax2.set_ylim(0, None)
    ax2.set_xlabel('odległość kamera–marker [m]')
    ax2.set_ylabel('błąd pozycji kostki [cm]')
    ax2.set_title('b) Pojedyncze detekcje (punkty) i mediany', fontsize=8, color=INK, loc='left')
    style(ax2)
    ax2.legend(frameon=False, fontsize=7, loc='upper left', labelcolor=INK)
    fig.tight_layout(pad=0.4, w_pad=1.2)
    fig.savefig(f'{out}.pdf')
    fig.savefig(f'{out}.png', dpi=300)
    print(f'zapisano {out}.pdf, {out}.png')


if __name__ == '__main__':
    main()
