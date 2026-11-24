# PyTorch CPU in the 10×10 training loop

The six-episode pilot uses the same training board, four held-out matches,
seed 17, batch size 32, learning rate 0.00001 and saved depth-10 Minimax
winner as the C++ pilot. Both DQN and hybrid (weight 0.5) start from fresh,
identically initialized models. During replay warm-up both use the available
transitions until the buffer contains 32. Validation and checkpointing run
after episode six.

The C++ trainer owns replay, gameplay, inference, exploration, target copies,
evaluation and checkpointing. A persistent Python process handles one batched
SGD update per learner decision. C++ samples its ordinary replay, packs the
observations, actions, rewards, masks and both networks' parameters, then sends
them through a local socket. It waits for the updated online parameters and
installs them before the next decision. This deliberately measures a simple
handoff with a full parameter round trip. The worker supports this pilot's
fixed network and board shape.

Play and update use the original pilot's steady-clock boundaries. Update
includes packing, both socket transfers, Python tensor construction, SGD,
weight installation and scheduled target copies. Loop includes checkpointing
and held-out evaluation. Process wall time also includes launching Python and
importing PyTorch. Runs execute sequentially with default CPU affinity on the
Ryzen 9 9950X in WSL (16 cores, 32 logical CPUs). Thread counts set PyTorch
intra-op parallelism. Each cell represents one six-episode run.

## Results

| Learner | Update backend | Play (s) | Update (s) | Loop (s) | Process (s) |
|---|---|---:|---:|---:|---:|
| DQN | C++ | 3.535 | 4.723 | 8.356 | 8.350 |
| DQN | PyTorch (1) | 3.629 | 0.521 | 4.252 | 5.110 |
| DQN | PyTorch (2) | 3.664 | 0.481 | 4.248 | 5.080 |
| DQN | PyTorch (4) | 3.602 | 0.446 | 4.153 | 5.030 |
| DQN | PyTorch (8) | 3.711 | 0.509 | 4.322 | 5.170 |
| DQN | PyTorch (16) | 3.917 | 0.518 | 4.539 | 5.380 |
| DQN | PyTorch (32) | 4.082 | 70.047 | 74.235 | 75.070 |
| Hybrid | C++ | 6.082 | 9.123 | 15.402 | 15.400 |
| Hybrid | PyTorch (1) | 6.344 | 0.888 | 7.435 | 8.280 |
| Hybrid | PyTorch (2) | 6.382 | 0.819 | 7.406 | 8.230 |
| Hybrid | PyTorch (4) | 6.380 | 0.801 | 7.385 | 8.250 |
| Hybrid | PyTorch (8) | 6.402 | 0.893 | 7.501 | 8.340 |
| Hybrid | PyTorch (16) | 6.731 | 0.930 | 7.866 | 8.700 |
| Hybrid | PyTorch (32) | 7.076 | 134.773 | 142.059 | 142.900 |

Four threads gave the shortest loop for both learners. Every backend reproduced
the C++ replay, update count, RNG states and evaluation. Across all final online
and target parameters the largest absolute difference was 3.1e-8 or less.
See results.txt for the raw process output.

## Reproduction

Use a fresh working copy of the original remote pilot's timed C++ source,
Makefile, tests, build objects and winner.txt. Copy its train executable to
train-cpp, and copy experiments/minimax-cpu-pilot and experiments/torch-full-loop
into the working copy. From that directory run:

```sh
bash experiments/torch-full-loop/run.sh
~/.venvs/chess-training-py312/bin/python experiments/torch-full-loop/compare.py
```

The remote directory is ~/training/jobs/snake-full-loop-321b3f0-b. Each run
saves a separate checkpoint. The comparison checks settings, epsilon, RNGs,
replay and update count exactly, and compares all online and target parameters.

## 32-thread contention check

Repeated both six-episode pilots sequentially with four-thread controls in
retry32-BobRL3wk. Each final checkpoint was byte-identical to the earlier
checkpoint at the same thread count. The original C++ timings are in the
table above.

| Learner | Threads / OpenMP wait policy | Play (s) | Update (s) | Loop (s) | Process (s) |
|---|---|---:|---:|---:|---:|
| DQN | 4 / default | 3.718 | 0.483 | 4.303 | 5.19 |
| DQN | 32 / default | 4.026 | 71.656 | 75.788 | 78.28 |
| DQN | 32 / passive | 3.610 | 9.424 | 13.135 | 14.01 |
| Hybrid | 4 / default | 6.470 | 0.817 | 7.493 | 8.33 |
| Hybrid | 32 / default | 7.125 | 136.893 | 144.234 | 145.16 |
| Hybrid | 32 / passive | 6.276 | 17.620 | 24.104 | 24.96 |

During the default-policy repeats, five-second WSL samples showed only the
worker and its trainer above the 1%-of-one-core threshold. The worker peaked
at 3198% (about 32 logical CPUs). Available Windows process counters showed
little activity, but the SSH account lacked system-wide Windows performance
counter access. There is no retrospective host trace for the first experiment.

Run bash experiments/torch-full-loop/retry32.sh in the completed job to repeat
the controls and save new checkpoints and WSL samples. For the passive runs
we set OMP_WAIT_POLICY=PASSIVE in the worker environment at 32 threads.
