#!/usr/bin/env bash
set -euo pipefail
# Run from the completed pilot directory. Each retry keeps its own checkpoints.
out=$(mktemp -d retry32-XXXXXXXX)
echo "retry_directory=$out"
export LC_ALL=C
top -b -w 120 -d 5 | awk '/^top -/ || /^%Cpu/ || (/^[ ]*[0-9]+ / && $9+0 >= 1) { print; fflush() }' > "$out/wsl-cpu.txt" &
monitor=$!
top_pid=$(jobs -p %%)
trap 'kill "$top_pid" "$monitor" 2>/dev/null || true' EXIT
p=experiments/minimax-cpu-pilot
py=~/.venvs/chess-training-py312/bin/python
for pair in dqn-32 dqn-4 hybrid-4 hybrid-32; do
    mode=${pair%-*}
    extra=()
    if [[ $mode == hybrid ]]; then extra=(--hybrid 0.5); fi
    echo "$pair $(date -u +%FT%TZ)"
    /usr/bin/time -f 'process_seconds=%e' "$py" experiments/torch-full-loop/worker.py "${pair#*-}" 6 \
        "$p/ten.txt" "$p/heldout.txt" "$out/$pair.model" --minimax-params winner.txt \
        --seed 17 --batch-size 32 --save-every 6 --evaluate-every 6 "${extra[@]}"
    cmp "$out/$pair.model" "$pair.model"
done
