# Fixed-batch DQN update benchmark

The fixture uses 32 distinct replay transitions sampled with seed 17 from the 166 in
the 10×10 DQN pilot checkpoint. Both networks start with the online and target
parameters from that checkpoint. Each implementation takes a float32 SGD step at
learning rate 0.00001 on the mean half-squared error for the chosen actions.

| Backend | Median update (ms) | 90th percentile (ms) |
|---|---:|---:|
| Production C++, g++ -O2 -DNDEBUG | 30.666 | 30.839 |
| PyTorch, four CPU threads | 1.524 | 1.639 |
| PyTorch ROCm, Radeon RX 7900 XT | 2.395 | 2.532 |

One remote run per backend, 10 warm-up updates and 50 timed updates after the
equivalence check. Each ROCm step copies both observations, actions, rewards and
masks from host memory and reads the scalar loss back before stopping the clock.
Weights and optimizer stay on device. PyTorch starts with a packed host batch.
C++ draws all 32 fixture entries through its normal replay sampler on each step.
Game simulation, Python batch construction, model loading and target-network
synchronization are outside the measurement. These are update timings, not episode timings.

For the first step, compare all four pre-update Q-values and the masked Bellman
target for each transition, allowing for replay order. Then compare every updated
online weight and bias with the saved C++ result. On both PyTorch backends the
maximum absolute Q-value or target difference was 7.46e-8 or less and the maximum
absolute parameter difference was 1.46e-11 or less. The largest error norm relative
to the C++ update norm for any parameter tensor was 8.34e-6 on CPU and 7.80e-6 on GPU.

## Reproduce on the training desktop

In an isolated working directory, copy src/, build/ and dqn-ten.model from the
remote pilot job snake-2e6276d-pilot-1123a. Copy this directory there as bench/
and run bash bench/run.sh. The script builds a separate executable with a trace
of the first C++ update and leaves the pilot checkpoint intact. The benchmark
uses the desktop's existing PyTorch 2.9.1 ROCm 7.2 environment. The completed job,
fixture, reference checkpoint, trace and raw C++ samples are in
~/training/jobs/snake-torch-321b3f0-a. The SHA-256 of batch.model was
5866f1e45fca1b81d66f3134201c518893ae357db8ac517d916766b3d71d48f7.
