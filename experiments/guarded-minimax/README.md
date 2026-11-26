# Guarded Minimax comparison

Both experiments pit the evaluated policy against the same tuned Minimax opponent from both seats. In the selector experiment, all three policies have the same eligible moves and fallback; five seeds measure uniform random selection. The final comparison restores the frozen hybrid's narrower 1.0-point gate and compares it with Minimax under the same 20-turn protection. Each row records both policies' outcome and turn count on the same start.

Run `python3 experiments/guarded-minimax/compare.py` for the paired totals. After building the tactical evaluator, add `--replay-layouts 2` to replay the first two layouts from both seats. Add `--model /path/to/ranked.model` to replay the narrow hybrid instead. The results files include the original decision timings and paired bootstrap intervals.

Guarded Minimax won 915 of 1,536 games and the frozen narrow hybrid won 890. Guarded Minimax gained 314 wins and surrendered 289. Its estimated advantage was 1.63 percentage points, with a paired 95% interval from −1.11 to +4.36.
