# Ranking unresolved successors

The scorer embeds a pair of successor boards with shared weights. Its three outputs predict losing, preserving or gaining a win relative to the frozen move. The inference rule chooses the alternative with the largest predicted gain minus loss probability and falls back to the frozen move when this difference is nonpositive.

To repeat the four training settings with CPU PyTorch and NumPy, run `python3 experiments/successor-scoring/train.py build/scorers`. Each seed uses 30 epochs, four threads and validation-only model selection. The direct-ranking term compares winning and losing alternatives at the same position. Opening-matched sampling targets rounds zero, one through two and later rounds with proportions 0.606, 0.306 and 0.088.

The compressed fixtures store one successor per line: game, round, action, outcome, frozen action, height, width, nine flattened board channels and 21 scalars. Outcome is measured after resuming the frozen continuation. The original training fixtures cover games 0–127. Additional opening positions, validation and test each use a separate split.

The paired position table records the initial scorer's decisions. The ablation choices table covers triggered games only; whole-game win rates use all 512 games. The confirmation table includes all 2,048 games for each locked sampling seed. Its seat pairs share a layout. The two results files preserve the desktop estimates and layout-level confidence intervals.
