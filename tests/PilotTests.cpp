/*************************************************************************
PilotTests - Exercises training, resume and play executables
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//- Implementation of the module <PilotTests> (file PilotTests.cpp) -

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <fstream>
#include <string>

using namespace std;

//------------------------------------------------------- Personal Include
#include "WorkflowTest.h"

//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
void testPilotResume()
// Algorithm : Compare split and uninterrupted training and load the archived model in the native player.
{
    string setup = "4 4\n3\n2 1\n0 0\n3 3\n";
    ofstream("starts.txt") << setup;

    // The test runner executes each case outside the repository root.
    string config = readFile(PILOT_CONFIG);
    string params = "experiments/minimax-cpu-pilot/winner.txt";
    config.replace(config.find(params), params.size(), PILOT_PARAMS);
    ofstream("settings.txt") << config;

    auto full = runExecutable({PILOT_EXECUTABLE, "4", "starts.txt", "starts.txt", "settings.txt", "full"});

    auto first = runExecutable({PILOT_EXECUTABLE, "2", "starts.txt", "starts.txt", "settings.txt", "split"});
    auto resumed = runExecutable({PILOT_EXECUTABLE, "2", "starts.txt", "starts.txt", "settings.txt", "split",
                                  "--resume", "split/episode-2"});

    check(full.first == 0 && first.first == 0 && resumed.first == 0, "Pilot training failed");

    // Both the learner and the pilot must reach the same state after four episodes.
    check(readFile("full/episode-4/model.model") == readFile("split/episode-4/model.model"),
          "Pilot resume changed the trained checkpoint");
    check(readFile("full/episode-4/pilot.state") == readFile("split/episode-4/pilot.state"),
          "Pilot resume changed the experiment metadata");

    auto player = runExecutable({SNAKE_EXECUTABLE, "dqn", "split/episode-4/model.model"}, setup);
    check(player.first == 0 && !player.second.empty(), "Native player could not use the pilot checkpoint");

    // Resume compares the saved settings byte for byte, including whitespace.
    ofstream("settings.txt", ios::app) << '\n';
    auto changed = runExecutable({PILOT_EXECUTABLE, "1", "starts.txt", "starts.txt", "settings.txt", "split",
                                  "--resume", "split/episode-4"});
    check(changed.first != 0, "Resume accepted different experiment settings");
}  //----- end of testPilotResume


vector<TestCase> getTests()
// Algorithm : Register the isolated pilot workflow test.
{
    return {{"pilot_resume", testPilotResume}};
}  //----- end of getTests
