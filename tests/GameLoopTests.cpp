/*************************************************************************
GameLoopTests - Executable protocol integration
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//- Implementation of the module <GameLoopTests> (file GameLoopTests.cpp) -

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <cerrno>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <sys/wait.h>
#include <unistd.h>

using namespace std;

//------------------------------------------------------- Personal Include
#include "Test.h"

//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
pair<int, string> playTranscript(const string &transcript)
// Algorithm : Run the real bot against a recorded protocol stream and capture its exit status and moves.
{
    ofstream input("input.txt");
    input << transcript;
    input.close();
    pid_t child = fork();
    check(child >= 0, "Cannot start bot process");

    if (child == 0)
    {
        alarm(10);

        if (!freopen("input.txt", "r", stdin) || !freopen("output.txt", "w", stdout) ||
            !freopen("error.txt", "w", stderr))
        {
            _exit(126);
        }

        execl(SNAKE_EXECUTABLE, SNAKE_EXECUTABLE, static_cast<char*>(nullptr));
        _exit(127);
    }

    int status;
    pid_t result;

    do
    {
        result = waitpid(child, &status, 0);
    } while (result < 0 && errno == EINTR);

    check(result == child && WIFEXITED(status), "Bot crashed or timed out");
    ifstream output("output.txt");
    string moves((istreambuf_iterator<char>(output)), istreambuf_iterator<char>());

    return {WEXITSTATUS(status), moves};
}  //----- end of playTranscript


void testDeathAndMoveNotifications()
// Algorithm : Verify collision notifications count once and repeated dead-player slots preserve the expected move sequence.
{
    string setup = "8 1\n3\n3 2\n0 0\n1 0\n7 0\n";

    for (const string failedMove : {"", "move 0 right\n"})
    {
        auto result = playTranscript(setup + "death 0\n" + failedMove +
                                     "move 2 left\ndeath 0\nmove 2 left\ndeath 0\ndeath 2\n");

        check(result.first == 0, "Bot rejected valid death and move notifications");
        check(result.second == "right\nright\nright\n", "Death messages changed the scheduled moves");
    }
}  //----- end of testDeathAndMoveNotifications


void testImmediateWinAndTrap()
// Algorithm : Play forced corridor positions to verify winning moves remain alive and trapped bots exit without hanging.
{
    auto win = playTranscript("3 1\n1\n2 1\n0 0\n2 0\ndeath 1\n");
    check(win.first == 0 && win.second == "right\n", "Winning move was not emitted correctly");

    auto trap = playTranscript("3 1\n1\n3 2\n0 0\n1 0\n2 0\ndeath 0\n");
    check(trap.first == 0 && trap.second.empty(), "Trapped player should close output without an invalid move");
}  //----- end of testImmediateWinAndTrap


void testEndOfInput()
// Algorithm : Verify EOF after setup ends cleanly while an incomplete setup produces an error.
{
    auto complete = playTranscript("8 1\n3\n3 2\n0 0\n1 0\n7 0\n");
    check(complete.first == 0 && complete.second.empty(), "EOF while awaiting an opponent did not exit cleanly");

    auto incomplete = playTranscript("8 1\n3\n3 2\n0 0\n");
    check(incomplete.first != 0 && incomplete.second.empty(), "Incomplete setup was accepted");
}  //----- end of testEndOfInput


void testInvalidProtocol()
// Algorithm : Reject malformed or out-of-order opponent messages before producing a move.
{
    string setup = "8 1\n3\n3 2\n0 0\n1 0\n7 0\n";

    for (const string event : {"move 0 diagonal\n", "move 2 left\n", "death 8\n"})
    {
        auto result = playTranscript(setup + event);
        check(result.first != 0 && result.second.empty(), "Invalid event was applied to the game");
    }
}  //----- end of testInvalidProtocol


vector<TestCase> getTests()
// Algorithm : Register executable-level protocol regression tests.
{
    return {
        {"death_and_move_notifications", testDeathAndMoveNotifications},
        {"immediate_win_and_trap", testImmediateWinAndTrap},
        {"end_of_input", testEndOfInput},
        {"invalid_protocol", testInvalidProtocol}
    };
}  //----- end of getTests
