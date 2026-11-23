/*************************************************************************
OptimiseTests - Exercises Minimax optimization and parameter loading
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//---- Implementation of the module <OptimiseTests> (file OptimiseTests.cpp) ----

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <fstream>
#include <string>
#include <vector>

using namespace std;

//------------------------------------------------------- Personal Include
#include "WorkflowTest.h"

//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
void testWinRateSelectsWinner()
// Algorithm : Check that the shorter winning candidate is saved over longer-surviving losers on a mixed-seat set.
{
    // With the default seed and three candidates, only the second candidate wins the first setup.
    // The losers last 19 turns there, the winner lasts 12, and every candidate loses the second setup in 2.
    ofstream("selection.txt") << "3 4\n2\n2 1\n1 0\n1 2\n"
                                 "3 1\n1\n2 2\n0 0\n2 0\n";

    auto first = runExecutable({OPTIMISE_EXECUTABLE, "selection.txt", "first.txt", "--population", "3",
                                "--generations", "1"});
    check(first.first == 0, "Initial generation failed: " + readFile("stderr.txt"));
    check(first.second.find("wins=1/2 win_rate=0.5 mean_survival_turns=7 ") != string::npos,
          "Selection must use wins and report the shorter survival time separately");

    auto later = runExecutable({OPTIMISE_EXECUTABLE, "selection.txt", "later.txt", "--population", "3",
                                "--generations", "3"});
    check(later.first == 0 && readFile("first.txt") == readFile("later.txt"),
          "Later generations must preserve the first best candidate on equal win rate");

    auto repeat = runExecutable({OPTIMISE_EXECUTABLE, "selection.txt", "repeat.txt", "--population", "3",
                                 "--generations", "3", "--seed", "5489"});
    check(repeat.first == 0 && readFile("repeat.txt") == readFile("later.txt"),
          "The explicit default seed should reproduce the selected parameters");
}  //----- end of testWinRateSelectsWinner


void testOptimizerOutputLoadsInPlayAndTraining()
// Algorithm : Save a winner from controlled matches and load the same artifact in Minimax play and Hybrid training and play.
{
    string setup = "3 1\n1\n2 1\n0 0\n2 0\n";
    ofstream("training.txt") << "4 1\n1\n2 1\n0 0\n3 0\n";
    ofstream("validation.txt") << setup << "3 1\n1\n2 2\n0 0\n2 0\n";
    auto search = runExecutable({OPTIMISE_EXECUTABLE, "validation.txt", "winner.txt", "--population", "4",
                                 "--generations", "3", "--seed", "17"});
    check(search.first == 0 && search.second.find("wins=1/2 win_rate=0.5 mean_survival_turns=2 ") != string::npos,
          "Optimizer must evaluate the supplied setups and seats");

    auto minimax = runExecutable({SNAKE_EXECUTABLE, "minimax", "--minimax-params", "winner.txt"}, setup);
    check(minimax.first == 0 && minimax.second == "right\n",
          "Minimax play must accept the saved winner and emit a legal move");

    auto training = runExecutable({TRAIN_EXECUTABLE, "1", "training.txt", "validation.txt", "hybrid.model",
                                    "--hybrid", "0", "--minimax-params", "winner.txt"});
    check(training.first == 0 && training.second.find("wins=1/2 win_rate=0.5 mean_survival_turns=2") != string::npos,
          "Hybrid training must accept the saved winner and finish evaluation");

    auto hybrid = runExecutable({SNAKE_EXECUTABLE, "hybrid", "hybrid.model", "0", "--minimax-params", "winner.txt"},
                                 setup);
    check(hybrid.first == 0 && hybrid.second == "right\n",
          "Hybrid play must accept the same parameter file");
}  //----- end of testOptimizerOutputLoadsInPlayAndTraining


void testExplicitParametersChangeDecisions()
// Algorithm : Compare default and shallow Minimax on a board where looking ahead changes the first move and game outcome.
{
    string setup = "3 2\n1\n2 1\n1 0\n0 0\n";
    ofstream("shallow.txt") << "0 0 0 0 0 0 0\n1\n";
    auto defaultPlay = runExecutable({SNAKE_EXECUTABLE}, setup);
    auto shallowPlay = runExecutable({SNAKE_EXECUTABLE, "minimax", "--minimax-params", "shallow.txt"}, setup);
    check(defaultPlay.first == 0 && defaultPlay.second == "down\n" &&
          shallowPlay.first == 0 && shallowPlay.second == "right\n",
          "Supplied weights and depth must reach the play policy");

    ofstream("training.txt") << "4 1\n1\n2 1\n0 0\n3 0\n";
    ofstream("validation.txt") << setup;
    auto defaultTrain = runExecutable({TRAIN_EXECUTABLE, "1", "training.txt", "validation.txt", "default.model",
                                       "--hybrid", "0"});
    auto shallowTrain = runExecutable({TRAIN_EXECUTABLE, "1", "training.txt", "validation.txt", "shallow.model",
                                       "--hybrid", "0", "--minimax-params", "shallow.txt"});
    check(defaultTrain.first == 0 && defaultTrain.second.find("wins=1/1") != string::npos &&
          shallowTrain.first == 0 && shallowTrain.second.find("wins=0/1") != string::npos,
          "Supplied parameters must reach training validation while omitted parameters use defaults");

    auto shallowHybrid = runExecutable({SNAKE_EXECUTABLE, "hybrid", "shallow.model", "0",
                                        "--minimax-params", "shallow.txt"}, setup);
    check(shallowHybrid.first == 0 && shallowHybrid.second == shallowPlay.second,
          "The Minimax branch of Hybrid play must also use supplied parameters");
}  //----- end of testExplicitParametersChangeDecisions


void testInvalidInputsPreserveOutputs()
// Algorithm : Reject an incomplete setup and invalid parameter files before changing outputs or emitting moves.
{
    ofstream("incomplete.txt") << "3 1\n1\n2 1\n0 0\n";
    ofstream("winner.txt") << "previous file";
    auto search = runExecutable({OPTIMISE_EXECUTABLE, "incomplete.txt", "winner.txt"});
    check(search.first != 0 && readFile("winner.txt") == "previous file",
          "A malformed setup must preserve the existing winner");

    for (const string params : {"0 0 0 0 0 0 0\n0\n", "0 0 0 0 0 0 0\n8 trailing\n", "0 0 0\n"})
    {
        ofstream("invalid.txt") << params;
        auto play = runExecutable({SNAKE_EXECUTABLE, "minimax", "--minimax-params", "invalid.txt"},
                                   "3 1\n1\n2 1\n0 0\n2 0\n");
        check(play.first != 0 && play.second.empty(), "Invalid parameters must fail before emitting moves");
    }

    auto training = runExecutable({TRAIN_EXECUTABLE, "1", "training.txt", "validation.txt", "new.model",
                                    "--minimax-params", "invalid.txt"});
    check(training.first != 0 && readFile("stderr.txt").find("Minimax parameters") != string::npos &&
          !ifstream("new.model"), "Invalid parameters must fail before training writes its checkpoint");
}  //----- end of testInvalidInputsPreserveOutputs


vector<TestCase> getTests()
// Algorithm : Register the optimizer and parameter workflow tests for the component runner.
{
    return {
        {"win_rate_selects_winner", testWinRateSelectsWinner},
        {"optimizer_output_loads_in_play_and_training", testOptimizerOutputLoadsInPlayAndTraining},
        {"explicit_parameters_change_decisions", testExplicitParametersChangeDecisions},
        {"invalid_inputs_preserve_outputs", testInvalidInputsPreserveOutputs}
    };
}  //----- end of getTests
