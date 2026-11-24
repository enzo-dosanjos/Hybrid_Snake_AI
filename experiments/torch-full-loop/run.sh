#!/usr/bin/env bash
set -euo pipefail
p=experiments/minimax-cpu-pilot
pilot=experiments/torch-full-loop
patch -p1 < "$pilot/worker.patch"
make -j8 CXXFLAGS='-O2 -DNDEBUG -std=c++17 -Wall' train > compile.log
py=~/.venvs/chess-training-py312/bin/python
for mode in dqn hybrid; do
    extra=()
    if [[ $mode == hybrid ]]; then extra=(--hybrid 0.5); fi
    for threads in cpp 1 2 4 8 16 32; do
        exe=(./train-cpp)
        if [[ $threads != cpp ]]; then exe=("$py" "$pilot/worker.py" "$threads"); fi
        name="$mode-$threads"
        echo "$name"
        /usr/bin/time -f 'process_seconds=%e' "${exe[@]}" 6 "$p/ten.txt" "$p/heldout.txt" "$name.model" \
            --minimax-params winner.txt --seed 17 --batch-size 32 --save-every 6 --evaluate-every 6 "${extra[@]}"
    done
done
