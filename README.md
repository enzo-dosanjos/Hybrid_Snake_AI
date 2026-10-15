# Hybrid AI for Competitive Snake

![Recorded Snake match with numbered heads, connected bodies, ringed tails and move metrics](docs/images/match-demo.gif)

*Illustrative hybrid match: two players grow every three rounds. An offensive
heuristic tries to cut off the opponent; six-ply Minimax checks survival and
rejects risky attacks. The animation shows which policy selected each move.

## Overview

![PROJECT UNDER REFACTORING — The test suite currently fails and the hybrid AI is not operational.](docs/images/refactoring-status.svg)

This is an advanced AI system for playing a multiplayer Snake games to participate in the [INSAlgo Snake Competition](https://github.com/INSAlgo/Concours-Snake)

The project implements a combination of 2 AI approaches in C++ without using any external libraries:
1. **Minimax Algorithm with Alpha-Beta Pruning**  
   *Focus: Defense & Survival*
  - **Self-preservation:** Avoids self-blocking by ensuring safe moves.
  - **Defensive Search:** Evaluates moves to trap or force the opponent into a dangerous position.
  - **Dynamic Evaluation:** Adjusts search based on board size and game state.

2. **Deep Q-Learning (DQN) with Convolutional & Fully Connected Neural Networks**  
   *Focus: Offense & Adaptability*
  - **Risk-Taking:** Learns to exploit opportunities and take calculated risks.
  - **Adaptive Strategy:** Adapts to opponent behavior and varying map sizes.
  - **Feature Extraction:** Uses a CNN to visually process the game board and an FCNN to make decisions based on combined features.

---

## Key Features

- **Hybrid AI Strategies:**
  - **Minimax with Alpha-Beta Pruning:** Guarantees defensive play and trap-setting.
  - **Deep Q-Network (DQN):** Combines CNN for board analysis with FCNN for decision making.
  - **Hybrid Integration:** Leverages the strengths of both approaches for optimal performance.

- **Advanced Spatial Risk Analysis:**
  - Dynamic risk heatmaps and multi-source flood fill algorithms for area control.
  - Temporal-spatial risk assessment to evaluate movement dangers.

- **Sophisticated Reward System:**
  - Rewards based on space control, kill zone detection, and anti-passivity.
  - Tailored metrics that adjust for board size and dynamic game scenarios.

- **Robust Training & Validation Tools:**
  - Experience replay buffer for stable training.
  - Tools for saving/loading models and performance analytics.
  - Integrated validation routines to analyze victories, survival turns, rewards, Q-value consistency, and action entropy.

---

## Architecture

The diagram below summarizes the target architecture for the refactoring.
The complete class interfaces, ownership relationships and behavioral contracts
are defined in [ClassDiagram.puml](ClassDiagram.puml).
Arrows show component dependencies.

```mermaid
flowchart TD
    subgraph Application
        GameLoop[GameLoop]
        TrainingLoop[TrainingLoop]
        InputHandler[InputHandler]
    end

    subgraph Game[Game state and rules]
        GameEngine[GameEngine]
        GameState[GameState]
        PlayerSelector[PlayerSelector]
        Observation[Observation]
        StateAnalyzer[StateAnalyzer]
    end

    subgraph Agents[Agent interface]
        DQNAgent[DQNAgent]
        MinimaxAgent[MinimaxAgent]
        HybridAgent[HybridAgent]
    end

    subgraph Search[Adversarial search]
        Minimax[Minimax]
        SpaceRiskAnalyzer[SpaceRiskAnalyzer]
    end

    subgraph Learning[Deep reinforcement learning]
        DQN[DQN]
        Networks[Online and target CNN + FCNN]
        ReplayBuffer[ReplayBuffer]
        Reward[Reward]
        ModelPersistence[ModelPersistence]
        ResultValidation[ResultValidation]
    end

    GameLoop --> InputHandler
    GameLoop --> GameEngine
    GameLoop --> Agents
    TrainingLoop --> GameEngine
    TrainingLoop --> Agents
    TrainingLoop --> DQN
    TrainingLoop --> Observation
    TrainingLoop --> StateAnalyzer
    TrainingLoop --> Reward
    TrainingLoop --> ModelPersistence
    TrainingLoop --> ResultValidation
    GameEngine --> GameState
    GameEngine --> PlayerSelector
    Observation --> GameState
    StateAnalyzer --> GameState
    DQNAgent --> Observation
    DQNAgent --> StateAnalyzer
    DQNAgent --> DQN
    HybridAgent --> Observation
    HybridAgent --> StateAnalyzer
    HybridAgent --> DQN
    HybridAgent --> Minimax
    MinimaxAgent --> Minimax
    Minimax --> SpaceRiskAnalyzer
    Minimax -->|Simulates independent copies| GameEngine
    DQN --> Networks
    DQN --> ReplayBuffer
    ModelPersistence --> DQN
    ResultValidation --> Agents
    ResultValidation --> GameEngine
```

- **GameEngine** owns the authoritative `GameState` and applies movement,
  collisions, growth, deaths and turn scheduling. Search uses copied states and
  the same transition rules as real play.
- **Observation and StateAnalyzer** transform a state into board channels and
  ordered numerical features. `EncodedState` groups these inputs with the legal
  action mask; `Transition` stores a learning experience.
- **Agents** share a `selectAction` interface. `DQNAgent` uses learned Q-values,
  `MinimaxAgent` uses search scores, and `HybridAgent` combines their normalized
  scores over legal actions using `(1 - lambda) * minimaxScore + lambda * qValue`.
- **DQN** owns online and target networks plus its replay buffer. CNN features
  pass through global average pooling and are combined with extra features before
  the FCNN produces four action values.
- **TrainingLoop** orchestrates episodes, rewards, replay training, target-network
  updates, model persistence and validation. Agents borrow the learner's DQN.

---

## Project Structure

```
├── include/
│   ├── CNN.h               # Convolutional neural network
│   ├── FCNN.h              # Fully connected neural network
│   ├── Minimax.h           # Minimax algorithm with alpha-beta pruning
│   ├── NNAI.h              # Combination of CNN and FCNN for AI
│   ├── ReplayBuffer.h      # Replay buffer for deep Q-Learning
│   ├── SpaceRiskAnalyzer.h # Spatial risk assessment
│   ├── Structs.h           # Data structures
│   └── Utils.h             # Utility functions
├── src/
│   ├── CNN.cpp               # Implementation of a convolutional neural network
│   ├── FCNN.cpp              # Implementation of a fully connected neural network
│   ├── Minimax.cpp           # Implementation of a minimax algorithm with alpha-beta pruning
│   ├── NNAI.cpp              # Implementation of the combination of CNN and FCNN for AI
│   ├── ReplayBuffer.cpp      # Implementation of a replay buffer for deep Q-Learning
│   ├── SpaceRiskAnalyzer.cpp # Implementation of a spatial risk assessment class
│   └── Reward.cpp             # Implementation of utility functions
```

---

## Game State Representation

`Observation::encode(state, perspective)` and
`StateAnalyzer::computeExtraFeatures(state, perspective)` consume a valid engine
snapshot and return owned values. Neither component retains or mutates a state.
Player IDs are `1..N`; the perspective is explicit and need not be the scheduled
player. Legal-action masks come from the engine snapshot.

### Board State

Layout is `[channel][y][x]`, with nine channels:

| Channel | Value |
| --- | --- |
| 0, 1, 2 | Perspective player's head, interior body and tail |
| 3, 4, 5 | All opponents' heads, interior bodies and tails |
| 6 | Unoccupied cell (not reachability) |
| 7 | Perspective head x / (W - 1), broadcast to all cells |
| 8 | Perspective head y / (H - 1), broadcast to all cells |

A one-segment snake is both head and tail. Dead snakes retain their occupied
channels. Coordinate channels are zero when the corresponding dimension is one.

### Extracted Features

A fixed player count produces
`StateAnalyzer::getFeatureCount(N) = 7 + 10*(N - 1)` values, independent of board
size, insertion order and deaths. Opponent slots always follow ascending player
ID with the perspective player omitted.

Concatenate these blocks in order:

| Block | Values, in order | Size |
| --- | --- | --- |
| Spatial, one block per opponent | Alive flag; own head to opponent segments: min, max, average distance; opponent head to own segments: min, max, average distance | 7*(N - 1) |
| Growth and size | M / 10; rounds until growth / M; own snake length / (W*H) | 3 |
| Progress | Capped elapsed scheduled slots / heuristic turn budget | 1 |
| Own status and schedule | Own alive flag; tanh(own normalized action entropy); (currentPlayer - 1) / (N - 1) | 3 |
| Opponent gameplay, one block per opponent | Legal-action count / 4; wall/corner proximity; tanh(normalized action entropy) | 3*(N - 1) |

For N=1, scheduled-player encoding is zero and opponent blocks are empty.
Both blocks for a dead opponent are entirely zero, including history metrics.
Its body still appears in the board tensor and blocks engine-calculated legal
moves. Changing a dead body's coordinates in a synthetic snapshot without
recomputing legal masks does not update those derived masks.

### Formulas and Conventions

- Distances retain the original Euclidean normalization `sqrt(W*W + H*H)` and
  include all stored segments (head and tail included).
- Growth uses the authoritative round, including dead-player slots. Rounds until
  growth are `(M - round % M) % M`: zero means the current round grows.
  `M / 10` retains the original scaling and can exceed one when M > 10.
- Elapsed slots are `round*N + currentPlayer - 1`. The original turn-budget
  heuristic is `W*H*M`, multiplied by 0.8 when M > 1. Progress is clamped to
  `[0,1]`; it does not determine termination or promise a maximum game duration.
- Entropy is `-sum(p*log2(p))/2`, giving zero for empty/repeated histories and one
  for equally distributed actions. Features retain the original `tanh` transform.
  Missing histories encode zero; engine histories hold the last five moves.
  Own history is zero when the perspective player is dead.
- Wall/corner proximity adds 0.5 for each axis within its outer quarter, then
  averages that weight with `1 - nearestWallDistance/(max(W,H)/2)`.
  Edge checks are symmetric and the result is clamped to `[0,1]`.

---

## Components

## SpaceRiskAnalyzer
SpaceRiskAnalyzer is a spatial risk assessment system for grid-based games that provides tools to evaluate spatial risks, 
available areas, and movement dangers in competitive grid environments like Snake.

### Features
- **Risk Heatmap Generation**: Calculates risk levels across the game grid based on how quickly different players can reach each cell
- **Space Accessibility Analysis**: Determines available movement space for each player using multi-source flood fill
- **Directional Risk Assessment**: Evaluates the danger of moving in specific directions
- **Efficient Caching**: Avoids redundant calculations by caching precomputed data

### Core Algorithms
- **Reach Timing Computation**: Uses BFS to calculate how many steps it takes each player to reach every cell
- **Multi-source Flood Fill**: Determines accessible areas and contested regions
- **Exponential Decay**: Applies decay to risk factors based on distance for more accurate risk assessment


### Minimax Algorithm
A classical AI approach designed for strategic defense.

- **Highlights:**
  - Alpha-beta pruning for efficient state search.
  - Evaluation functions that integrate spatial risk analysis.
  - Adaptive to varying board sizes and dynamic game conditions.

### Neural Network AI
Deep reinforcement learning approach:
- **CNN**: Processes the board state visually : 3x3 conv → maxpool → 3x3 conv
- **Feature Fusion**: Concatenates spatial + analytic features
- **FCNN**: Makes decisions based on extracted features and the output of the CNN : 64 → 32 → 4 FC layers with ReLU except the last one which is a linear layer
- **ReplayBuffer**: Implements experience replay for stable learning
- Training mechanisms with decay exploration rates


## Usage
The main.cpp file provide 3 different main functions:
- `gameMain` to play a game with the combination of both AI
- `trainingMain` to train the neural network AI
- `optimiseMain` to optimise the Minimax parameters


## todo:
- finish main.cpp
- finish optimise.cpp
- finish train.cpp

- optimise terminal state detection in Minimax (multiple methods and don't use the alive state)
- optimise Minimax using transition table

- 📈 performance benchmark
| Metric                | Minimax Only | DQN Only | Hybrid AI |
|-----------------------|--------------|----------|-----------|
| Win Rate (1v1)        | 0%           | 0%       | **0%**    |
| Decisions/sec         | 0            | 0        | 0         |
| Training Convergence  | N/A          | 0h       | 0h        |
