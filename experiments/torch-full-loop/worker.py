"""Run the ordinary trainer with a synchronous CPU PyTorch batch-update worker."""
import os
import socket
import subprocess
import sys

import numpy as np
import torch
from torch import nn


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(9, 16, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(16, 32, 3, padding=1), nn.ReLU())
        self.dense = nn.Sequential(nn.Linear(53, 64), nn.ReLU(), nn.Linear(64, 32),
                                   nn.ReLU(), nn.Linear(32, 4))

    def forward(self, state):
        board = state[:, :900].reshape(-1, 9, 10, 10)
        return self.dense(torch.cat((self.conv(board).mean((2, 3)), state[:, 900:]), 1))

    def wire_parameters(self):
        # C++ stores weights input-first, PyTorch stores weights output-first.
        for parameter in self.parameters():
            yield parameter.transpose(0, 1) if parameter.ndim > 1 else parameter


def read_values(stream, count):
    data = stream.read(count * 4)
    if len(data) != count * 4:
        raise EOFError("Incomplete batch from C++")
    return torch.from_numpy(np.frombuffer(data, dtype=np.float32).copy())


def main():
    torch.set_num_threads(int(sys.argv[1]))
    online, target = Network(), Network()
    actions = torch.arange(4)
    parent, child = socket.socketpair()
    with parent, child, subprocess.Popen(
        ['./train', *sys.argv[2:]], pass_fds=(child.fileno(),),
        env=dict(os.environ, SNAKE_WORKER_FD=str(child.fileno()))
    ) as trainer:
        child.close()
        try:
            with parent.makefile('rb') as stream:
                while header := stream.read(12):
                    count, rate, gamma = np.frombuffer(header, dtype=np.float32)
                    with torch.no_grad():
                        for net in (online, target):
                            for parameter in net.wire_parameters():
                                parameter.copy_(read_values(stream, parameter.numel()).reshape_as(parameter))
                    batch = read_values(stream, int(count) * 1845).reshape(-1, 1845)
                    action, reward, mask = batch[:, -3:].T
                    with torch.no_grad():
                        legal = (mask.long()[:, None] >> actions) & 1
                        future = target(batch[:, 921:1842]).masked_fill(~legal.bool(), -torch.inf).amax(1)
                        expected = reward + float(gamma) * torch.where(mask != 0, future, 0.)
                    q = online(batch[:, :921]).gather(1, action.long()[:, None]).flatten()
                    online.zero_grad(set_to_none=True)
                    ((q - expected).square().mean() * .5).backward()
                    with torch.no_grad():
                        for p in online.parameters():
                            p.add_(p.grad, alpha=-float(rate))
                        updated = torch.cat([p.reshape(-1) for p in online.wire_parameters()])
                    parent.sendall(updated.numpy().tobytes())
        finally:
            parent.close()
        if trainer.wait():
            raise SystemExit(trainer.returncode)


if __name__ == '__main__':
    main()
