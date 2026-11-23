# Tests

## GitHub Actions

`.github/workflows/tests.yml` runs `make all` and `make test` on Ubuntu for
pushes to `main` or `refactoring` and pull requests targeting either branch.
The workflow also supports manual runs. Build failures and failing suites fail
the CI job.

## Local execution

Run from the repository root on a POSIX system with GNU Make and a C++17 compiler.

```sh
make test
make test TEST=game-engine
make test TEST=input-handler
make test TEST=game-loop
make test TEST=cnn
make test TEST=minimax
make test TEST=--list
make test TEST=all
```

After a suite builds, run an individual case directly.

```sh
./build/tests/cnn pooling
```

`RunTests.cpp` selects and builds suites. `TestMain.cpp` runs each named test in
its own process, with a 20-second timeout and a temporary working directory.
Exceptions, assertions, crashes and timeouts fail the test without stopping the
other cases. Compilation failures fail the overall run and are reported separately.

| Suite | Coverage |
| --- | --- |
| GameEngineTests | Masks across rectangular boards, growth over multiple rounds/rates, self/opponent/wall collisions, death idempotency, dead turn slots, copied successors and seeded complete games. |
| InputHandlerTests | Configuration and origins, typed events, player ID normalization, malformed input, EOF and exact move output. |
| GameLoopTests | Executable protocol transcripts, paired death/move notifications, dead slots, trapped players, EOF and invalid event order. |
| ObservationTests | Every channel/cell against state, both perspectives, one-cell dimensions, bodies/tails, growth and tail release, snapshot independence. |
| StateAnalyzerTests | Stable opponent ordering under all insertion permutations, retained dead-player slots, zeroed dead-player metrics and finite features on narrow boards. |
| SpaceRiskAnalyzerTests | Per-player arrival slots, contested territory, directional risk, lookahead and static occupancy, retained dead bodies and rectangular boards. |
| MinimaxTests | Forced moves, no-action cases, state isolation, all 30 ordered starts on a 3×2 board compared with exhaustive terminal outcomes, two/three-player complete games. |
| CNNTests | Multi-channel rectangular convolutions with stride/padding, analytical pooling values, multi-layer numerical weight/bias gradients, reset and weight copying. |
| FCNNTests | Independent forward calculations, numerical gradients for every parameter, first-layer learning, two-sample accumulation, reset, loss reduction and independent weight copying. |
| ReplayBufferTests | Small/full capacities, exact batch sizes, empty sampling, transition ownership, seeded sampling and sample diversity. |
| RewardTests | Observed eliminations, terminal wins and losses, trapped players, pressure and reduced escapes across a decision interval. |
| DQNTests | All 16 legal masks, exploration/greedy selection, terminal targets independent of next state, learning from a real engine episode, reproducible replay updates. |
| DQNAgentTests | Scheduled-player encoding, shared learner state and exploration mode. |
| HybridAgentTests | Legal-score normalization, configured mixing, equal scores, tie order and shared exploration. |
| PersistenceTests | Predictions on multiple states, training-step restoration, exploration/replay/target-network continuation, rejected invalid files without changing the model. |
| ResultValidationTests | Evaluated seats and opponent lineups, survival and win accounting, dead slots and evaluation without exploration. |
| TrainingLoopTests | Executed learner action, intervening opponent turns, terminal rewards, empty next masks and illegal actions. |
| TrainingCliTests | Training and resumption equivalence, Hybrid training, DQN and Hybrid checkpoint play, and invalid input preserving checkpoints. |
| OptimiseTests | Candidate ranking, parameter output consumed by play and training, move changes under explicit weights, and invalid input preserving outputs. |

## Test contracts

Failed persistence loads must report an exception and preserve the existing model.
Replay sampling takes up to the requested batch size without replacement and
returns an empty batch when there are no stored transitions.

The exhaustive search reference uses engine transitions but does not use Minimax
scoring or pruning. Engine tests independently check the transition rules.
Numerical references use copied layer values to check forward and backward
computations. The executable tests use small fixed boards and protocol
transcripts to exercise the application workflows.
