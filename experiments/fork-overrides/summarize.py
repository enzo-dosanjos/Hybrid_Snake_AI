"""Summarize the archived target audit, match controls and learned override forks."""

import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

root = Path(__file__).resolve().parent


def rows(filename):
    with (root / filename).open() as stream:
        return list(csv.DictReader(stream))


audit = rows("target-audit.csv")
vetoed = [float(row["target_change"]) for row in audit if int(row["vetoed"])]
print(f"Highest legal Q vetoed: {len(vetoed)}/{len(audit)}, mean target change on veto: {mean(vetoed):.4f}")

matches = defaultdict(dict)
for row in rows("matches.csv"):
    group = row["study"], row["policy"], row["seed"]
    game = int(row["game"])
    assert game not in matches[group] and int(row["seat"]) == 1 + game % 2
    matches[group][game] = int(row["win"])

for group, games in sorted(matches.items()):
    assert set(games) == set(range(128))
    print(*group, sum(games.values()), "/ 128")

print("Decisive overrides:", Counter(int(row["win_change"]) for row in rows("override-comparison.csv")))
