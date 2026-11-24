"""Check paired final checkpoints, including the replay and both RNG states."""
from pathlib import Path

import numpy as np


for mode in ('dqn', 'hybrid'):
    ref = Path(f'{mode}-cpp.model').read_text().splitlines()
    for threads in (1, 2, 4, 8, 16, 32):
        got = Path(f'{mode}-{threads}.model').read_text().splitlines()
        assert got[:11] == ref[:11], (mode, threads, 'config, update count, epsilon or RNG')
        assert got[21:] == ref[21:], (mode, threads, 'replay')
        largest = 0.
        for row in range(11, 21):
            a = np.fromstring(ref[row], sep=' ')
            b = np.fromstring(got[row], sep=' ')
            np.testing.assert_allclose(a, b, rtol=0., atol=2e-6)
            largest = max(largest, abs(a-b).max())
        print(mode, threads, 'identical replay and RNG; max parameter difference', largest)
