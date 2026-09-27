# Tests

## GitHub Actions

`.github/workflows/tests.yml` runs `make test` on every push to `main` or
`refactoring`, and on pull requests targeting either branch. Runs involving
`refactoring` also execute checkpoints 1, 2 and 3 as independent jobs, so one
checkpoint failure does not cancel the others. The workflow can also be run
manually from GitHub Actions.

The branch name is exactly `refactoring`. If another name is chosen, update both
the branch filters and the checkpoint job condition in the workflow.
Commit the workflow, `tests/Makefile`, both C++ runners and all component test
files so they are available in GitHub's clean checkout.

Failures and compilation blockers fail CI, including unfinished refactoring
interfaces. Completing stages 1–4 alone does not make checkpoint 1 pass: it also
includes Observation and StateAnalyzer. No deployment is configured.

## Local execution

Run from the repository root:

```sh
make test
make test TEST=game-engine
make test TEST=input-handler
make test TEST=game-loop
make test TEST=cnn
make test TEST=minimax
make test TEST=--list
```

After a suite builds, run an individual case directly:

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
| StateAnalyzerTests | Stable opponent ordering under all insertion permutations, retained dead-player slots, zeroed dead-player metrics, entropy properties and finite features on narrow boards. |
| MinimaxTests | Forced moves, no-action cases, state isolation, all 30 ordered starts on a 3×2 board compared with exhaustive terminal outcomes, two/three-player complete games. |
| CNNTests | Multi-channel rectangular convolutions with stride/padding, analytical pooling values, multi-layer numerical weight/bias gradients, reset and weight copying. |
| FCNNTests | Independent forward calculations, numerical gradients for every parameter, first-layer learning, two-sample accumulation, reset, loss reduction and independent weight copying. |
| ReplayBufferTests | Small/full capacities, exact batch sizes, empty sampling, transition ownership, seeded sampling and sample diversity. |
| DQNTests | All 16 legal masks, exploration/greedy selection, terminal targets independent of next state, learning from a real engine episode, reproducible replay updates. |
| PersistenceTests | Predictions on multiple states, training-step restoration, exploration/replay/target-network continuation, rejected invalid files without changing the model. |

Test files contain named functions grouped by component. `checkpointSuites()` is a temporary grouping that will be
used during the refactoring phase. It is in
`RunTests.cpp`, available through:

```sh
make test TEST="--checkpoint 1"
make test TEST="--checkpoint 2"
make test TEST="--checkpoint 3"
```

todo : Delete that function and its argument branch when the refactoring is complete;

## API alignment

Game, agent, feature and learning tests use the target API from
`ClassDiagram.puml`. Later agent, feature and learning suites remain build-blocked
until their interfaces exist.
Neural-network tests use the existing public layer-building and inspection APIs;

Failed persistence loads must report an exception and preserve the existing model.
Replay sampling returns an empty batch when the buffer is empty, and fills the
requested size when sufficient samples exist.

The exhaustive search reference uses engine transitions but does not use Minimax
scoring or pruning; engine tests separately check the rules. Numerical references
use copied layer values and never call production forward/backward methods.
These tests cover behavior and numerical correctness, not learning convergence,
full competition orchestration or performance benchmarking. The game-loop suite
checks subprocess communication through fixed protocol transcripts.
