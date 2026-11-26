"""Summarize paired games and optionally replay each seat on the first N layouts."""
import argparse
import csv
from pathlib import Path
import subprocess

directory = Path(__file__).resolve().parent
root = directory.parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--replay-layouts', type=int, default=0)
parser.add_argument('--model', type=Path, help='replay the narrow hybrid using this ranked checkpoint')
args = parser.parse_args()

with (directory / 'final-guard-comparison.csv').open() as stream:
    games = list(csv.DictReader(stream))

layouts = {}
gained = surrendered = 0
seat_wins = [[0, 0], [0, 0]]
for game in games:
    layout = int(game['layout'])
    seat = int(game['seat'])
    origins = tuple(game[key] for key in ('width', 'height', 'growth', 'p1x', 'p1y', 'p2x', 'p2y'))
    prior, seats = layouts.setdefault(layout, (origins, set()))
    assert prior == origins and seat not in seats
    seats.add(seat)

    minimax = int(game['minimax_win'])
    hybrid = int(game['hybrid_win'])
    seat_wins[0][seat - 1] += minimax
    seat_wins[1][seat - 1] += hybrid
    gained += minimax > hybrid
    surrendered += minimax < hybrid

    if layout < args.replay_layouts:
        command = [root / 'build/experiments/Evaluate']
        policy = 'minimax'
        if args.model:
            command.append(args.model.resolve())
            policy = 'hybrid'
        width, height, growth, x, y, a, b = origins
        setup = f'{width} {height}\n{growth}\n2 {seat}\n{x} {y}\n{a} {b}\n'
        result = subprocess.check_output(command, cwd=root, text=True, input=setup)
        expected = [game[policy + '_win'], game[policy + '_turns']]
        if result.split() != expected:
            raise ValueError(f'layout {layout}, seat {seat}: expected {expected}, got {result}')

assert len(layouts) == 768 and all(seats == {1, 2} for _, seats in layouts.values())
print('Minimax and hybrid seat wins:', seat_wins)
print('Minimax gains and losses:', gained, surrendered)
