"""Train the four paired successor scorers on the archived fork outcomes."""
from collections import defaultdict
from pathlib import Path
import argparse
import copy
import gzip

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def load(split):
    """Pair each candidate successor with the frozen successor from the same turn."""
    states = {}
    groups = defaultdict(list)
    with gzip.open(Path(__file__).parent / 'fixtures' / (split + '.txt.gz'), 'rt') as stream:
        for line in stream:
            game, turn, action, win, baseline, height, width, data = line.split(' ', 7)
            values = np.fromstring(data, sep=' ', dtype=np.float32)
            board = values[:-21].reshape(9, int(height), int(width))
            moves = states.setdefault((int(game), int(turn)), {})
            moves[int(action)] = (int(baseline), board, values[-21:], int(win))

    for (game, turn), moves in states.items():
        for action, (baseline, board, features, win) in moves.items():
            if action == baseline:
                continue
            _, frozen_board, frozen_features, frozen_win = moves[baseline]
            groups[board.shape[-2:]].append((
                frozen_board, frozen_features, board, features, win - frozen_win + 1,
                1 / (len(moves) - 1), game, turn, action, baseline, win, frozen_win
            ))

    return {shape: tuple(torch.from_numpy(np.stack(column)) for column in zip(*rows))
            for shape, rows in groups.items()}


class Scorer(nn.Module):

    def __init__(self):
        super().__init__()
        self.register_buffer('mean', ex.mean(0))
        self.register_buffer('std', ex.std(0).clamp(min=0.01))
        self.conv = nn.Sequential(nn.Conv2d(9, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())
        self.enc = nn.Sequential(nn.Linear(48 + 21, 48), nn.ReLU())
        self.preference = nn.Sequential(nn.Linear(96, 32), nn.ReLU(), nn.Linear(32, 1))
        self.tie = nn.Linear(48, 1)

    def embed(self, b, e):
        z = self.conv(b)
        return self.enc(torch.cat([z.mean((2, 3)), (z * b[:, 0:1]).sum((2, 3)), (z * b[:, 4:5]).sum((2, 3)), (e - self.mean) / self.std], 1))

    def forward(self, b, e, c, f):
        a, b = (self.embed(b, e), self.embed(c, f))
        gain = (self.preference(torch.cat([a, b], 1)) - self.preference(torch.cat([b, a], 1))).flatten()
        return torch.stack([-gain, self.tie((a - b).abs()).flatten(), gain], 1)


def meta(data):
    out = {}
    for k, v in data.items():
        by = defaultdict(list)
        for i, gr in enumerate(zip(v[6].tolist(), v[7].tolist())):
            by[gr].append(i)
        loser = torch.full((len(v[4]), 2), -1)
        weight = torch.zeros(len(v[4]))
        for idx in by.values():
            wins = [i for i in idx if v[10][i]]
            ls = [i for i in idx if not v[10][i]]
            if wins and ls:
                for i in wins:
                    loser[i, :len(ls)] = torch.tensor(ls)
                    weight[i] = 1 / (len(ls) * len(wins))
        out[k] = (loser, weight)
    return out

def predictions(model, data):
    pos = defaultdict(list)
    with torch.no_grad():
        for v in data.values():
            out = model(*v[:4]).softmax(1)
            pref = (out[:, 2] - out[:, 0]).tolist()
            for score, g, r, a, b, y, y0 in zip(pref, *[t.tolist() for t in v[6:12]]):
                pos[g, r].append((score, a, y, b, y0))
    rows = []
    for (g, r), items in sorted(pos.items()):
        score, a, y, b, y0 = max(items, key=lambda x: (x[0], -x[1]))
        rand = sum((z[2] for z in items)) / len(items)
        rows.append((g, r, y0, rand, y, y if score > 0 else y0, int(score > 0), a if score > 0 else b))
    return np.asarray(rows)

def metrics(arr):
    base, rand, alt, pick = arr[:, 2:6].T
    mix = (rand > 0) & (rand < 1)
    return ((alt[mix] - rand[mix]).mean(), (pick - base).mean())

def train(name, opening, direct, seed):
    data = expanded if opening else original
    aux = meta(data)
    torch.manual_seed(seed)
    model = Scorer()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    best = None
    for ep in range(1, 31):
        batches = []
        for k, v in data.items():
            n = len(original[k][4])
            if opening:
                r = v[7]
                phase_masks = (r == 0, (r > 0) & (r <= 2), r > 2)
                prob = torch.zeros(len(r), dtype=torch.double)
                for mask, fraction in zip(phase_masks, [0.606, 0.306, 0.088]):
                    prob[mask] = fraction / v[5][mask].sum()
                ix = torch.multinomial(prob, n, replacement=True)
            else:
                ix = torch.randperm(n)
            batches.extend(((k, idx) for idx in ix.split(64)))
        for bi in torch.randperm(len(batches)):
            k, idx = batches[bi]
            v = data[k]
            logits = model(*(t[idx] for t in v[:4]))
            w = v[5][idx].float()
            loss = (F.cross_entropy(logits, v[4][idx], reduction='none') * w).sum() / w.sum()
            if direct:
                neg, importance = aux[k]
                ii, j = torch.where(neg[idx] >= 0)
                if len(ii):
                    loseridx = neg[idx[ii], j]
                    lo = model(*(t[loseridx] for t in v[:4])).softmax(1)
                    hi = logits[ii].softmax(1)
                    diff = hi[:, 2] - hi[:, 0] - (lo[:, 2] - lo[:, 0])
                    w2 = importance[idx[ii]]
                    loss = loss + (F.softplus(-4 * diff) * w2).sum() / (4 * w2.sum())
            opt.zero_grad()
            loss.backward()
            opt.step()
        m = metrics(predictions(model, val))
        key = (sum(m), m[1], -ep)
        if best is None or key > best:
            best = key
            state = copy.deepcopy(model.state_dict())
            selected = ep
    torch.save(state, d / f'{name}-{seed}.pt')
    print('selected', name, seed, selected, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    d = args.output
    d.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(4)
    original = load('original')
    opening = load('opening')
    for group in opening.values():
        group[6].add_(10000)

    expanded = {k: tuple(torch.cat([x, y]) for x, y in zip(v, opening[k]))
                for k, v in original.items()}
    val = load('val')
    ex = torch.cat([v[1] for v in original.values()] + [v[3] for v in original.values()])
    test = load('test')
    for seed in (717, 41, 79):
        for name, sampling, ranking in [('current', False, False), ('ranking', False, True),
                                        ('sampling', True, False), ('both', True, True)]:
            train(name, sampling, ranking, seed)
            model = Scorer()
            model.load_state_dict(torch.load(d / f'{name}-{seed}.pt', weights_only=True))
            print(name, seed, 'ranking and win differences on triggered positions',
                  metrics(predictions(model, test)))
