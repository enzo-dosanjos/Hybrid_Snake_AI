# Match viewer

One browser view for live testing and recorded matches. The visualizer calls
`snake.game()` and records its actual board; it does not implement another set
of movement or collision rules. Python 3.10+ is required. The viewer itself uses
only the standard library and browser APIs.

## Live games

From the repository root:

```sh
# Seeded random-legal demo; no C++ build needed
python3 tools/visualizer/Viewer.py random random

# Play against your compiled bot
make AI
python3 tools/visualizer/Viewer.py human ./AI

# Watch two bot processes
python3 tools/visualizer/Viewer.py ./AI ./AI
```

Open **http://127.0.0.1:8765**. The server binds only to localhost. Use the arrow
keys or direction buttons on your turn; only currently legal moves are accepted.
A turn token prevents a delayed or double click from applying to the next turn.
Two to four players are supported, with corner starts. Options include
`--grid 16 12`, `--growth 3`, `--delay 0.2`, `--port 8766` and `--output path.json`.
`random` and `human` are reserved player names; use `./random` for an executable
with that name.

The game continues while reviewing history. **Follow live** returns to the latest
frame; replay controls pause playback, not the match. Human controls are available
only while following live. The server stays open after the match for inspection.
Ctrl+C stops the session and saves the recorded frames.

A bot must speak the competition protocol. This viewer does not fix or replace
unfinished C++ agents. `random` is a demo baseline, not Minimax or DQN.

## Metrics and visual conventions

- A stable color and number identify each player. Heads point along the last
  movement; connected segments show body order; rings mark tails. Initially head
  and tail share one cell and no movement direction has been established.
- Eliminated snakes are dimmed and retain their occupied cells.
- Player cards show **post-move** length, legal moves and accessible space.
- Move details show the chosen action, **pre-move** legal alternatives, length
  change, outcome and observed response time.
- Space is the count of reachable empty cells with every body held fixed. It
  excludes the head, can overlap between players, and is not a territory score
  or a prediction of future survival.
- Response time measures the runner's wait for the move, including communication
  and human thinking. It is not an internal search or neural-network benchmark.
- Turns count scheduled slots, including dead players; rounds start at zero.
  Growth follows the competition runner. The round displayed belongs to the next
  scheduled turn, while move details describe the preceding move.

## Replays and sampled games

The completed match is saved to `replays/latest.json` by default. **Save replay**
downloads the frames currently received by the browser, including an unfinished
match. Open `tools/visualizer/index.html` directly in a browser and choose
**Open replay** to inspect a saved file without starting a server.

```sh
# Record without a browser or artificial delay
python3 tools/visualizer/Viewer.py ./AI ./AI --headless --delay 0

# Save only matches 100, 200, ... of a batch
python3 tools/visualizer/Viewer.py ./AI ./AI --headless --delay 0 \
    --games 1000 --record-every 100 --output replays/evaluation.json
```

Batch files are named `evaluation-000100.json`, `evaluation-000200.json`, etc.
Each match launches fresh bot processes. **This is evaluation recording, not a
training loop:** no learning or model checkpoint loading is added here. Sampling
controls files saved; metrics are still collected during every match.

When the C++ training loop is implemented, it can export the same version 1 JSON
once every 100 episodes. The browser does not depend on the producer. The
`Recorder.observe()` implementation defines the format:

- Root: `version`, `width`, `height`, `growth`, `names`, `frames`, `finished`, `error`.
- Frame: completed slot count `turn`, next-turn `round`, ordered `players`, `event`.
- Player: one-based `id`, `alive`, head-first `body` coordinates `[x,y]`, `legal`
  direction strings and `space`.
- Event: one-based `player`, `action` (or null), `outcome`, `decisionMs` (or null),
  `legalBefore`, `lengthChange`. Initial frame uses `event: null`.

Minimax/DQN scores and decision explanations are intentionally deferred until the
agents expose real values. No inferred or simulated AI scores are displayed.
