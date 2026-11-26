# One-second decision deadline

The desktop measurement ran the two C++ players as separate processes on Linux. The referee used the competition board and measured elapsed wall time from delivering input to receiving a complete reply. It started up to eight games at a time on separate physical cores. A deliberately late reply checked that the watchdog actually fired. A trapped player ending its output was recorded separately.

The archive includes all 1,536 games from the final comparison and 72 stress games covering 10×10 with growth every five rounds and 14×14 with growth every three or ten rounds. Guarded Minimax returned 108,357 legal moves. The slowest took 386.1788 ms on the desktop. No response missed the deadline. These measurements cover the sampled two-player boards and machine; the search itself has a depth bound but no wall-clock cutoff.

From the project root:

```sh
python3 experiments/guarded-minimax/deadline/check.py
make -f experiments/guarded-minimax/deadline/Makefile build/experiments/DeadlinePlayer
python3 experiments/guarded-minimax/deadline/measure.py build/deadline-repeat
python3 experiments/guarded-minimax/deadline/check.py build/deadline-repeat
```

Use a new output directory for each repetition. Add `--limit 2` to the measurement command for a two-seat smoke test. The included adapter reuses the repository proof search. The desktop timing archive came from the original experimental executable.
