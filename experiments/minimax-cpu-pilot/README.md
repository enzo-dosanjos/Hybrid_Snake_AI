# Remote CPU pilot

Source: merged main at 2e6276d. The remote job and all three model checkpoints are in
~/training/jobs/snake-2e6276d-pilot-1123a on the training desktop's Ubuntu WSL instance.
The production trainer was unchanged. run.sh applies timing.patch to a fresh disposable
source copy and builds with g++ -O2 -DNDEBUG. Run it from that copy's repository root.

Six candidates, three generations, seed 17, eight tuning matches against default Minimax.
Four layouts each played from both seats, growth interval three. The winner scored 6/8
and had depth 10. In the four held-out matches (two new layouts, both seats), the winner
and the default both scored 2/4 against default Minimax. No evidence of improvement yet.

All profiles used seed 17, batch size 32, default learning rate and the saved winner as
the Minimax opponent. For the additional 10x10 runs use six episodes, ten.txt as the
training setup, heldout.txt for evaluation, --save-every 6 --evaluate-every 6 and a
fresh output filename per run. Hybrid additionally used --hybrid 0.5 and the same winner
in the learner's search. Evaluation ran once at the end of each run.

| Board and behavior | Episodes | Updates | Play seconds | Update seconds | Loop seconds |
|---|---:|---:|---:|---:|---:|
| 6x5 DQN | 12 | 155 | 0.382327 | 1.323 | 1.79427 |
| 10x10 DQN | 6 | 166 | 3.46746 | 4.63694 | 8.20316 |
| 10x10 Hybrid | 6 | 308 | 6.06436 | 9.09085 | 15.3526 |

Times are elapsed on the desktop using steady_clock. Play includes encoding, acting,
opponent search, engine moves and reward. Update includes replay insertion/sampling,
online and target forward passes, backpropagation and scheduled target copies.
Loop also includes checkpoint writes and validation, and excludes program startup.
Each configuration ran once. The GPU-accelerable fraction is smaller than the whole
update fraction because replay handling and transfers still cost time.
