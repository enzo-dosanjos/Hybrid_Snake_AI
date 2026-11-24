"""Benchmark the pilot's version-1 checkpoint and default Snake network only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['prepare', 'cpu', 'cuda'])
parser.add_argument('checkpoint', type=Path)
args = parser.parse_args()
lines = args.checkpoint.read_text().splitlines()
assert lines[0] == 'HybridSnakeDQN 1'
assert lines[1] == '2 2 2 21'
assert lines[4:8] == ['2', 'ReLU 3 1 1 16', 'ReLU 3 1 1 32', '2 64 32']
# Header, step/epsilon, two RNG states, then five layers each for online and target.
first = 11
count = first + 10
if args.mode == 'prepare':
    n = int(lines[count])
    idx = np.random.default_rng(17).choice(n, 32, replace=False)
    fixture = lines[:count] + ['32']
    for i in idx:
        fixture.extend(lines[count+1+3*i:count+4+3*i])
    Path('batch.model').write_text('\n'.join(fixture)+'\n')
    print('fixture', n, idx.tolist(), hashlib.sha256(Path('batch.model').read_bytes()).hexdigest())
    quit()

import time
import torch
from torch import nn

device = torch.device(args.mode)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
if device.type == 'cuda':
    assert torch.version.hip and torch.cuda.is_available()
print(torch.__version__, torch.version.hip, torch.cuda.get_device_name() if device.type == 'cuda' else '4 CPU threads', flush=True)

class Network(nn.Module):
    def __init__(self, rows):
        super().__init__()
        self.layers = nn.ModuleList([nn.Conv2d(9,16,3,padding=1), nn.Conv2d(16,32,3,padding=1),
                                     nn.Linear(53,64), nn.Linear(64,32), nn.Linear(32,4)])
        with torch.no_grad():
            for layer, row in zip(self.layers, rows):
                values = torch.from_numpy(np.fromstring(row, sep=' ', dtype=np.float32))
                out, inp, *kernel = layer.weight.shape
                layer.weight.copy_(values[:-out].reshape(inp,out,*kernel).transpose(0,1))
                layer.bias.copy_(values[-out:])

    def forward(self, board, extras):
        for layer in self.layers[:2]:
            board = layer(board).relu()
        x = torch.cat((board.mean((2,3)), extras), dim=1)
        for layer in self.layers[2:-1]:
            x = layer(x).relu()
        return self.layers[-1](x)

def state(row):
    data = np.fromstring(row, dtype=np.float32, sep=' ')
    assert tuple(data[:2]) == (10,10)
    return data[2:902].reshape(9,10,10), data[902:]

columns = [[] for _ in range(7)]
for i in range(32):
    header, now, successor = lines[count+1+3*i:count+4+3*i]
    action, done, mask, next_mask, reward = header.split()
    fields = (*state(now), *state(successor), int(action), float(reward),
              [bool(int(next_mask) & (1 << a)) and not int(done) for a in range(4)])
    for col, value in zip(columns, fields):
        col.append(value)
host = [torch.as_tensor(np.asarray(col)) for col in columns]
host[5] = host[5].float()
online = Network(lines[first:first+5]).to(device)
target = Network(lines[first+5:first+10]).to(device)
opt = torch.optim.SGD(online.parameters(), lr=1e-5)
gamma = float(lines[3].split()[1])

def update():
    b,e,nb,ne,a,r,m = (t.to(device) for t in host)
    with torch.no_grad():
        future = target(nb,ne).masked_fill(~m, -torch.inf).amax(1)
        y = r + gamma * torch.where(m.any(1), future, 0.)
    q = online(b,e)
    loss = (q.gather(1,a[:,None]).flatten()-y).square().mean() * .5
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return q, y, loss.item()  # scalar device-to-host transfer and synchronization

q, y, _ = update()
ref = np.loadtxt('trace.txt', dtype=np.float32)
actual = torch.cat((q.detach(), y[:,None]),1).cpu().numpy()
# C++ shuffles the same 32 transitions. Match complete rows including targets.
remaining = list(ref)
worst = 0.
for row in actual:
    j = np.argmin(np.max(np.abs(np.array(remaining)-row), axis=1))
    expected = remaining.pop(j)
    np.testing.assert_allclose(row, expected, atol=2e-5, rtol=2e-5)
    worst = max(worst, float(abs(expected-row).max()))
expected = Network(Path('after.model').read_text().splitlines()[first:first+5])
initial = Network(lines[first:first+5])
errors = []
for got, want, old in zip(online.parameters(), expected.parameters(), initial.parameters()):
    delta = want.detach()-old.detach()
    err = got.detach().cpu()-want.detach()
    errors.append((float(err.abs().max()), float(err.norm()/delta.norm())))
    assert errors[-1][0] < 2e-6 and errors[-1][1] < .01, errors
for _ in range(10):
    update()
ms = []
for _ in range(50):
    start = time.perf_counter()
    update()
    ms.append(1000*(time.perf_counter()-start))
print(json.dumps(dict(device=args.mode, max_q_target_error=worst, max_weight_error=max(x[0] for x in errors),
                      worst_relative_layer_update_error=max(x[1] for x in errors),
                      median_ms=np.median(ms), p90_ms=np.percentile(ms,90))))
