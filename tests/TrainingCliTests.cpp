/*************************************************************************
TrainingCliTests - Exercises training, resume and play executables
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//- Implementation of the module <TrainingCliTests> (file TrainingCliTests.cpp) -

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <fstream>
#include <string>

using namespace std;

//------------------------------------------------------- Personal Include
#include "WorkflowTest.h"

//---------------------------------------------------------------- PRIVATE
static void writeSetups()
// Algorithm : Write a one-decision training board and a separate two-seat validation set.
{
    ofstream("training.txt") << "4 1\n1\n2 1\n0 0\n3 0\n";
    ofstream("validation.txt") << "3 1\n1\n2 1\n0 0\n2 0\n"
                                  "3 1\n1\n2 2\n0 0\n2 0\n";
}  //----- end of writeSetups


//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
void testResumeMatchesUninterruptedTraining()
// Algorithm : Compare fresh and resumed checkpoint bytes and verify the update-based synchronization cadence.
{
    writeSetups();
    auto full = runExecutable({TRAIN_EXECUTABLE, "2", "training.txt", "validation.txt", "full.model",
                               "--target-every", "2", "--save-every", "1"});
    check(full.first == 0, "Uninterrupted training failed: " + readFile("stderr.txt"));
    check(full.second ==
          "checkpoint episode=1 updates=1\n"
          "checkpoint episode=2 updates=2\n"
          "evaluation episode=2 updates=2 wins=1/2 win_rate=0.5 mean_survival_turns=2\n",
          "Training must report periodic and final checkpoints and the held-out match metrics");

    auto first = runExecutable({TRAIN_EXECUTABLE, "1", "training.txt", "validation.txt", "split.model",
                                "--target-every", "2", "--evaluate-every", "1"});
    check(first.first == 0 && first.second.find("evaluation episode=1 updates=1") != string::npos,
          "The first training segment did not update and evaluate");
    auto resumed = runExecutable({TRAIN_EXECUTABLE, "1", "training.txt", "validation.txt", "split.model",
                                  "--resume", "--target-every", "2", "--evaluate-every", "1", "--seed", "17"});
    check(resumed.first == 0 && resumed.second.find("checkpoint episode=1 updates=2") != string::npos,
          "Resume must continue the saved update counter using the restored learner");
    check(readFile("full.model") == readFile("split.model"),
          "Training state must match after resume despite the extra evaluation and different initial seed");

    auto unsynced = runExecutable({TRAIN_EXECUTABLE, "2", "training.txt", "validation.txt", "unsynced.model",
                                    "--target-every", "3"});
    check(unsynced.first == 0 && readFile("full.model") != readFile("unsynced.model"),
          "The checkpoint must contain the target update scheduled at step two");
}  //----- end of testResumeMatchesUninterruptedTraining


void testTrainHybridAndPlayCheckpoint()
// Algorithm : Train a Hybrid policy and exercise both checkpoint-backed policies on protocol input.
{
    writeSetups();
    auto training = runExecutable({TRAIN_EXECUTABLE, "1", "training.txt", "validation.txt",
                                   "hybrid.model", "--hybrid", "0.5"});
    check(training.first == 0, "Hybrid training failed: " + readFile("stderr.txt"));

    auto dqn = runExecutable({SNAKE_EXECUTABLE, "dqn", "hybrid.model"},
                             "3 1\n1\n2 1\n0 0\n2 0\ndeath 1\n");
    check(dqn.first == 0 && dqn.second == "right\n",
          "DQN play must load the checkpoint and emit a legal protocol move");
    auto hybrid = runExecutable({SNAKE_EXECUTABLE, "hybrid", "hybrid.model", "0"},
                                "3 2\n1\n2 1\n1 0\n0 0\nmove 1 down\ndeath 1\n");
    check(hybrid.first == 0 && hybrid.second == "down\nright\n",
          "Hybrid play at zero neural weight must follow the winning Minimax branch");

    auto incompatible = runExecutable({SNAKE_EXECUTABLE, "dqn", "hybrid.model"},
                                      "4 1\n1\n3 1\n0 0\n2 0\n3 0\n");
    check(incompatible.first != 0 && incompatible.second.empty(),
          "An incompatible checkpoint must be rejected before emitting protocol output");
}  //----- end of testTrainHybridAndPlayCheckpoint


void testBadTrainingInputPreservesCheckpoint()
// Algorithm : Reject accidental overwrite and malformed validation before writing training state.
{
    writeSetups();
    ofstream("existing.model") << "previous checkpoint";
    auto overwrite = runExecutable({TRAIN_EXECUTABLE, "1", "training.txt", "validation.txt", "existing.model"});
    check(overwrite.first != 0 && readFile("existing.model") == "previous checkpoint",
          "Fresh training must preserve an existing checkpoint");

    ofstream("validation.txt", ios::app) << "3 1\n1\n2 2\n";
    auto invalid = runExecutable({TRAIN_EXECUTABLE, "1", "training.txt", "validation.txt", "new.model"});
    check(invalid.first != 0 && !ifstream("new.model"),
          "An incomplete validation setup must fail before training saves a checkpoint");
}  //----- end of testBadTrainingInputPreservesCheckpoint


vector<TestCase> getTests()
// Algorithm : Register executable integration regressions for training and play.
{
    return {
        {"resume_matches_uninterrupted_training", testResumeMatchesUninterruptedTraining},
        {"train_hybrid_and_play_checkpoint", testTrainHybridAndPlayCheckpoint},
        {"bad_training_input_preserves_checkpoint", testBadTrainingInputPreservesCheckpoint}
    };
}  //----- end of getTests
