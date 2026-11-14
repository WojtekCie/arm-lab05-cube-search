"""Klient akcji /search_cubes: wysyla cel, wypisuje postep i porownuje wynik z ground truth.

Uzycie:  ros2 run cube_search search_client [--expected 5] [--coverage 0.95] [--timeout 1800]
"""
import argparse
import math
import os
import re
import sys
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.utilities import remove_ros_args

from cube_search_interfaces.action import SearchCubes

GT_RE = re.compile(r'Aruco\s+(\d+):\s*x:\s*([-+0-9.eE]+)\s+y:\s*([-+0-9.eE]+)')


def load_ground_truth(path):
    gt = {}
    try:
        with open(path) as f:
            for line in f:
                m = GT_RE.search(line)
                if m:
                    gt[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    except OSError:
        pass
    return gt


def print_result(res, gt):
    print('\n================ WYNIK AKCJI /search_cubes ================')
    print(f'success={res.success}  pokrycie={res.coverage:.1%}  czas={res.elapsed:.0f} s (sym.)')
    print(f'komunikat: {res.message}')
    frame = res.cubes[0].position.header.frame_id if res.cubes else '-'
    print(f'\n ID |  x [m]   y [m]  (ramka {frame}) | obs | rozrzut [cm] |'
          + ('  GT x    GT y  | blad [cm]' if gt else ''))
    errors = []
    for c in sorted(res.cubes, key=lambda c: c.id):
        p = c.position.point
        line = (f'{c.id:3d} | {p.x:7.3f} {p.y:7.3f}                  | {c.observations:3d} |'
                f' {100 * c.std_dev:10.1f}   |')
        if c.id in gt:
            gx, gy = gt[c.id]
            err = math.hypot(p.x - gx, p.y - gy)
            errors.append(err)
            line += f' {gx:7.3f} {gy:7.3f} | {100 * err:8.1f}'
        elif gt:
            line += '   (brak ID w ground truth)'
        print(line)
    if gt:
        missing = sorted(set(gt) - {c.id for c in res.cubes})
        print(f'\nznalezione: {len(res.cubes)}/{len(gt)}; nieznalezione ID: {missing or "-"}')
        if errors:
            print(f'blad pozycji: sredni {100 * sum(errors) / len(errors):.1f} cm, '
                  f'maks. {100 * max(errors):.1f} cm')


def main(argv=None):
    argv = sys.argv if argv is None else argv
    parser = argparse.ArgumentParser(description='Klient akcji /search_cubes')
    parser.add_argument('--expected', type=int, default=5,
                        help='zakoncz po potwierdzeniu tylu kostek (0 = tylko kryterium pokrycia)')
    parser.add_argument('--coverage', type=float, default=0.95, help='wymagane pokrycie (0..1)')
    parser.add_argument('--timeout', type=float, default=1800.0, help='limit czasu [s] symulacji')
    parser.add_argument('--ground-truth',
                        default=os.path.expanduser('~/.ros/arm05_sim/logs/latest.txt'))
    args = parser.parse_args(remove_ros_args(argv)[1:])

    rclpy.init(args=argv)
    node = Node('cube_search_client')
    client = ActionClient(node, SearchCubes, 'search_cubes')
    if not client.wait_for_server(timeout_sec=30.0):
        print('Serwer akcji /search_cubes niedostepny.')
        return 1

    goal = SearchCubes.Goal()
    goal.expected_cubes = args.expected
    goal.coverage_goal = args.coverage
    goal.timeout = args.timeout
    last = [0.0]

    def on_feedback(msg):
        f = msg.feedback
        if time.monotonic() - last[0] >= 5.0:
            print(f'[{f.state:8s}] pokrycie {f.coverage:6.1%}  potwierdzone ID: {list(f.found_ids)}',
                  flush=True)
            last[0] = time.monotonic()

    print(f'Wysylam cel: expected_cubes={goal.expected_cubes}, coverage_goal={goal.coverage_goal}, '
          f'timeout={goal.timeout:.0f} s', flush=True)
    send_future = client.send_goal_async(goal, feedback_callback=on_feedback)
    rclpy.spin_until_future_complete(node, send_future)
    handle = send_future.result()
    if handle is None or not handle.accepted:
        print('Cel odrzucony.')
        return 1
    result_future = handle.get_result_async()
    try:
        rclpy.spin_until_future_complete(node, result_future)
    except KeyboardInterrupt:
        print('Anulowanie celu...')
        rclpy.spin_until_future_complete(node, handle.cancel_goal_async(), timeout_sec=5.0)
        rclpy.spin_until_future_complete(node, result_future, timeout_sec=15.0)
    wrapped = result_future.result()
    if wrapped is None:
        print('Brak wyniku.')
        return 1
    print_result(wrapped.result, load_ground_truth(args.ground_truth))
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
