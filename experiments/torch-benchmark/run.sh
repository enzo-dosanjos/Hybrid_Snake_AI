#!/usr/bin/env bash
set -euo pipefail
# In a fresh copy of the pilot source/build directory, with this benchmark in bench/.
py=~/.venvs/chess-training-py312/bin/python
"$py" bench/bench.py prepare dqn-ten.model
# Trace the actual first production batch, before SGD, in the benchmark-only object.
python3 - <<'PY'
from pathlib import Path
s = Path('src/DQN/DQN.cpp').read_text()
needle = '        int action = static_cast<int>(transition.action);'
assert s.count(needle) == 1
s=s.replace(needle,'''        static int traced = 0;
        if (traced++ < 32) {
            for (float v : values) std::cerr << std::setprecision(9) << v << ' ';
            std::cerr << target << '\\n';
        }
'''+needle)
Path('DQN-trace.cpp').write_text('#include <iomanip>\n#include <iostream>\n'+s)
PY
g++ -O2 -DNDEBUG -std=c++17 -I. -Isrc/DQN bench/Reference.cpp DQN-trace.cpp $(find build/src -name '*.o' ! -name DQN.o) -o reference
./reference > cpp-times.txt 2> trace.txt
"$py" -c "import numpy as n; a=n.loadtxt('cpp-times.txt',usecols=1);print('C++ median/p90 ms',n.percentile(a,[50,90]))"
"$py" bench/bench.py cpu batch.model
"$py" bench/bench.py cuda batch.model
