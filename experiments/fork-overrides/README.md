# Fork outcomes and the 1.0-point override margin

Run `python3 experiments/fork-overrides/summarize.py` from the repository root.
The board fixtures contain consecutive pairs of games with the same origins and
opposite learner seats. Game numbers are zero-based within each study.

The target audit inspected nonterminal replay transitions. It recorded whether
the highest-Q legal continuation fell outside the 1.0-point Minimax margin and
the resulting change in the discounted bootstrap target. The target-study match
rows compare three standard-target checkpoints and frozen demonstration weights
with five uniform-eligible seeds against the tuned opponent on 64 layouts.

For the ranking study, collect decisions along frozen demonstration-hybrid games
on the 160 collection layouts from both seats. At each position with multiple
eligible moves, fork each move once and resume the same frozen hybrid against
tuned Minimax until a snake dies. Group outcomes by game and round. Use the
first 128 layouts for training and the last 32 for validation. At positions
with both outcomes, minimize softplus of the losing score minus the winning
score, weighting each position equally. Fit a copy with Adam, rate 0.0001,
batch size 32, four CPU threads and seed 1717. Select by validation loss across
20 epochs. The per-epoch selection log records which epochs improved that loss.

The ranking-study match rows evaluate that copy, frozen demonstration weights
and the random controls on a separate 64 layouts from both seats. The SHA-256
manifest identifies the two external checkpoint files.

The later one-override study trained on 64 layouts and selected a threshold on
24 layouts. Branches compared a single outside-margin move with the frozen
move, then resumed the 1.0 hybrid. On 64 further layouts the policy intervened
33 times. The diagnosis CSV covers all 16 interventions that changed the winner.
Its depth-20 labels describe forced outcomes under adversarial replies.
