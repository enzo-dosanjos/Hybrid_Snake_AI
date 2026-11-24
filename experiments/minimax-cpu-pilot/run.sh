#!/usr/bin/env bash
set -eu
p=experiments/minimax-cpu-pilot
patch -p1 < "$p/timing.patch"
flags='-O2 -DNDEBUG -std=c++17 -Wall'
make -j4 CXXFLAGS="$flags" train optimise > build.log
./optimise "$p/tuning.txt" winner.txt --population 6 --generations 3 --seed 17
cat winner.txt
# Use the optimizer's match reader and the same validation implementation for the held-out comparison.
sed '/^int main(int argc/,$d' optimise.cpp > evaluate-pilot.cpp
cat >> evaluate-pilot.cpp <<'CPP'
int main(int argc, char **argv)
{
    MinimaxAgent opponent(MinimaxParams{});
    MinimaxAgent candidate(argc == 3 ? loadMinimaxParams(argv[2]) : MinimaxParams{});
    auto matches = readValidationSet(argv[1], opponent);
    auto result = ResultValidation().validate(candidate, matches);
    cout << "heldout " << (argc == 3 ? "winner" : "default") << " wins=" << result.wins << "/" << result.matches << " survival=" << result.meanSurvivalTurns << endl;
}
CPP
g++ $flags evaluate-pilot.cpp $(find build/src/GameEngine -name '*.o') build/src/Agent/MinimaxAgent.o build/src/Minimax/*.o build/src/DQN/ResultValidation.o -o evaluate-pilot
./evaluate-pilot "$p/heldout.txt"
./evaluate-pilot "$p/heldout.txt" winner.txt
./train 12 "$p/training.txt" "$p/heldout.txt" pilot.model --seed 17 --minimax-params winner.txt --batch-size 32 --save-every 12 --evaluate-every 12
