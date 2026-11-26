# Can a DQN improve competitive Snake decisions beyond Minimax?

We started with a division of labor: Minimax would keep a snake alive, and a DQN would recognize opportunities to trap the opponent. The practical question was whether giving the network a choice could turn more losses into wins than wins into losses within the competition's one-second decision budget.

**We keep guarded Minimax as the two-player reference.** Against tuned Minimax it recorded 25 more wins than the frozen hybrid across 1,536 matched games. The uncertainty on that difference includes zero. The reason to keep it is that it achieved the higher observed score, needs only the search parameters, and met the measured deadline. The learned selector's early gains did not establish a repeatable advantage on fresh layouts.

## What counted as an improvement?

Each policy faced the same opponent from the same origins and both seats. A **gain** means the candidate won a game the reference lost on that layout and seat. A **surrender** means the reverse. This distinction matters because two policies with similar win rates can change the winner in hundreds of games. We treated a layout and its two seats as a unit for the final confidence interval. Survival turns were diagnostic; a longer game that still ended in defeat counted as a loss. Different studies used different layouts, so totals below are compared within each study.

A *fork* asks a narrower question. From a recorded decision, play one candidate action and then resume the same frozen policy against the same opponent until elimination. This isolates the consequence of that first move under the specified continuation. A *proof* asks whether a player can force the outcome against every reply within a fixed number of scheduled player turns. A winning fork can remain unresolved by the proof search. Neither label alone demonstrates that the DQN has learned a general trap.

## Establishing the opponents and training the learner

We tuned the territory, blocking, heat-risk and length weights alongside search depth. Six candidates over three generations yielded a depth-10 configuration that won six of eight tuning games. On the four initial held-out games, the winner and the default each won two against default Minimax. We froze the tuned parameters to give subsequent learners a consistent opponent. The small tuning panel alone offers little evidence about the optimal search settings. [Tuning design and CPU pilot](../experiments/minimax-cpu-pilot/README.md)

The DQN takes a board encoding and numerical state features and outputs one Q-value per direction. We began with direct training against Minimax, compared the episode-800 and episode-1,000 checkpoints, and moved to a mixed-opponent pilot with varied starts, alternating seats, slower exploration and terminal rewards (+1 for winning, −1 for losing). Episode 717 became the frozen pilot reference. Continued training and a curriculum tested whether more updates would help. Minimax demonstrations then supplied a stronger starting policy for imitation and fine-tuning. Freezing the demonstration checkpoint let us test reward settings and move selection without changing the baseline between comparisons. [Pilot settings](../experiments/terminal-pilot/pilot.conf) · [Frozen checkpoint identities](../experiments/fork-overrides/models.sha256)

We also checked the cost of running those experiments. On a 10×10 batch of 32, one update took a median 30.666 ms in the original C++, 1.524 ms in four-thread PyTorch CPU and 2.395 ms on ROCm including device transfers. Updated parameters agreed with C++ within 1.46 × 10⁻¹¹ in the compatibility check. In a six-episode DQN pilot, the four-thread CPU worker reduced loop time from 8.356 to 4.153 seconds including packing and handoff. This motivated the checkpoint-compatible LibTorch CPU trainer. Faster updates made experiments cheaper; playing strength was evaluated separately. [One-update benchmark](../experiments/torch-benchmark/README.md) · [Full-loop comparison](../experiments/torch-full-loop/README.md)

## Did the DQN choose useful attacks?

The weighted hybrid combined normalized search and Q-values. A later rule gave the network a more specific job: pick among moves within **1.0 raw Minimax point** of the best. We tested uniform random selection over the same moves to see how much a learned preference added beyond varying the play.

The training target also deserved scrutiny. DQN bootstrapped from the largest legal next-action Q-value, even if the hybrid would veto that action. In 359 of 1,000 audited nonterminal transitions the top legal move fell outside the margin; the mean target change on those vetoes was 0.0772. On 128 fresh games the three standard-target hybrids won 64, 62 and 61; the demonstration hybrid won 60, and random-eligible seeds ranged from 56 to 64. The audit established a mismatch. Those full-game results gave little reason to trust Q-value ordering by itself. [Target audit and random controls](../experiments/fork-overrides/README.md)

We then trained on forks with both a winning and a losing eligible move, teaching a copy of the demonstration model to rank the winner higher. On a separate 128 games, the ranked hybrid won **69**, the frozen demonstration hybrid **58**, and five random seeds averaged **60**. That justified keeping `ranked.model` for the next comparison. Allowing learned exceptions outside the margin was harder: a single-override policy intervened 33 times, gaining eight wins and surrendering eight. It found genuinely useful moves, but handed away just as many outcomes. [Fork collection, ranking and override diagnosis](../experiments/fork-overrides/README.md)

## What did the 20-turn guard contribute?

The guard spends its horizon on the candidate move and up to 19 subsequent player turns. It keeps immediate terminal decisions and sole legal moves. Otherwise it prefers proven wins, then unresolved moves, retaining the preferred action when eligible. If that action is excluded, it takes the eligible move with the best Minimax score. If all moves lose it keeps the original selection. An unresolved move can still be dangerous beyond the horizon. [Guard implementation and rules](../experiments/tactical-guard/README.md)

| Policy against tuned Minimax, same 512 games | Wins | Gained / surrendered vs. preceding row |
| --- | ---: | ---: |
| Earlier direct-search hybrid | 258 | — |
| Frozen hybrid, applying the guard every turn | 322 | 64 / 0 |
| Same guard with a trained selector for unresolved moves | 314 | 83 / 91 |

The repeatable improvement on this panel came from applying tactical protection throughout play. Learning which unresolved move to take changed 174 outcomes against the protected frozen hybrid, for eight fewer wins overall. That motivated comparing successor positions directly. [Paired game records](../experiments/tactical-guard/unresolved-matches.csv)

## Could a successor scorer learn when to intervene?

A shared encoder compared the frozen and alternative successor boards. The scorer predicted whether switching would gain, preserve or surrender a win, falling back to the frozen move when predicted gain failed to exceed predicted loss. On 500 held-out positions it gained **57** wins and surrendered **59**. We tested two explanations: the objective might need to rank alternatives directly, or the training sample might underrepresent opening decisions. [Scorer design and fixtures](../experiments/successor-scoring/README.md)

In the four-way ablation, opening-matched sampling alone added 11, 12 and 11 wins across three seeds on 512 games. We advanced that setting to 1,024 fresh layouts played from both seats. The frozen policy won 1,210 of 2,048 games. The three scorers won 1,220, 1,202 and 1,236. Their mean advantage was **0.46 percentage points** (paired 95% interval **−1.51 to +2.39**). On positions with both winning and losing alternatives, their mean ranking advantage over random was **0.89 points** (interval **−1.41 to +3.13**). Opening-matched sampling gave a promising development result; confirmation left both advantages uncertain. [Four-way ablation](../experiments/successor-scoring/ablation-results.txt) · [Independent confirmation](../experiments/successor-scoring/confirmation-results.txt)

## Which policy should choose when search delegates?

First we kept the guard, eligible set, opponent and fallback identical and changed only the selector. On the same 512 games, tuned Minimax won **298**, frozen ranked selection **186**, and uniform random selection averaged **184.8** over five seeds. When the DQN could choose across this broader set, its decisions performed much like random selection. The narrow hybrid still deserved its own test because its 1.0-point margin excludes many moves the proof search leaves unresolved. [Controlled selector comparison](../experiments/guarded-minimax/selector-results.txt)

For the final test, guarded Minimax and the frozen **narrow** hybrid each played tuned Minimax on 768 fresh layouts from both seats. These are paired results against a common opponent.

| Final policy | Seat 1 | Seat 2 | Total wins | Win rate |
| --- | ---: | ---: | ---: | ---: |
| Guarded Minimax | 422/768 | 493/768 | 915/1,536 | 59.57% |
| Frozen narrow hybrid | 395/768 | 495/768 | 890/1,536 | 57.94% |

Guarded Minimax gained **314** wins and surrendered **289**. The net advantage of **1.63 percentage points** had a layout-paired 95% interval of **−1.11 to +4.36**. Seat 1 accounted for the observed lead; the hybrid won two more games from seat 2. The experiment supports retaining the simpler reference while leaving open a small strength advantage in either direction. [Paired results](../experiments/guarded-minimax/paired-results.txt) · [Game-level data](../experiments/guarded-minimax/final-guard-comparison.csv)

## Could the reference meet the one-second deadline?

The Linux referee ran two separate C++ processes and timed input delivery through receipt of the complete reply. The watchdog caught a deliberately late response. We repeated all 1,536 final games and added 72 stress games on 10×10 and 14×14 boards. All final winners and game lengths matched the archive. Across the 1,608 games, **all 108,357 legal guarded replies arrived within one second**. The maximum was **386.18 ms**. Another 653 guarded turns ended with a trapped player exiting without a legal move. [Protocol, stress cases and timing archive](../experiments/guarded-minimax/deadline/README.md)

The evidence covers two-player games against the tuned opponent, the tested board sizes and the desktop used for measurement. The proof search has a depth limit but no internal wall-clock cutoff. A different opponent pool, larger board or slower competition host would require a new evaluation. Within the conditions we tested, guarded Minimax combines the strongest observed final score with measured deadline compliance and no neural checkpoint to train, distribute or select. That is the reference future policies should beat.

## Reproduce the reported totals

From the repository root, these commands summarize the archived matches and verify the deadline evidence:

```sh
python3 experiments/fork-overrides/summarize.py
python3 experiments/guarded-minimax/compare.py
python3 experiments/guarded-minimax/deadline/check.py
```

The README animation is a separate head-to-head exhibition between the finalists. Guarded Minimax starts at (3, 3) and the frozen ranked hybrid starts at (9, 7) on a 12×11 board with growth every three rounds. Both use the 20-turn guard. Of six exhibition games, we selected this one because the hybrid played nine moves that differed from the top Minimax move and the guard intervened once. Guarded Minimax won on turn 198. The paired results above use the common tuned opponent. [Watch the match](../README.md)
