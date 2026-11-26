"""Verify both seats' game results and count guarded replies against the one-second limit."""
import argparse
from collections import Counter
import csv
import gzip
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory', nargs='?', type=Path, default=Path(__file__).resolve().parent)
args = parser.parse_args()
with (args.directory / 'games.csv').open() as stream:
    games = {int(row['game']): row for row in csv.DictReader(stream)}

times = []
statuses = Counter()
turns = Counter()
archive = args.directory / 'turns.csv.gz'
stream = gzip.open(archive, 'rt') if archive.exists() else (args.directory / 'turns.csv').open()
with stream:
    for row in csv.DictReader(stream):
        game = int(row['game'])
        assert int(row['turn']) == turns[game] and game in games
        turns[game] += 1
        if row['status'] not in ('move', 'trapped_eof'):
            raise ValueError(f'Unexpected protocol response: {row}')
        if row['guarded'] == '1':
            statuses[row['status']] += 1
            if row['status'] == 'move':
                times.append(float(row['ms']))

for game, row in games.items():
    assert turns[game] == int(row['turns']) and row['end'] == 'trapped_eof'
    if row['group'] == 'final':
        assert (row['win'], row['turns']) == (row['old_win'], row['old_turns'])

print('Games verified:', len(games))
print('Guarded replies:', dict(statuses))
print('Maximum legal decision time (ms):', max(times))
if max(times) >= 1000:
    raise ValueError('The guarded player exceeded the competition time limit')
