/*************************************************************************
InputHandlerTests - Configuration and event parsing
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//- Implementation of the module <InputHandlerTests> (file InputHandlerTests.cpp) -

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <sstream>

using namespace std;

//------------------------------------------------------- Personal Include
#include "Test.h"
#include "src/GameEngine/InputHandler.h"

//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
void testConfigurationAndOrigins()
// Algorithm : Read a complete setup and verify that configuration and origins retain their one-based identities.
{
    istringstream input("8 3\n2\n3 2\n0 0\n4 1\n7 2\nmove 0 right\n");
    auto config = InputHandler::readGameConfig(input);
    auto origins = InputHandler::readPlayersOrigins(config.N, input);

    check(config == GameConfig{8, 3, 2, 3, 2}, "Configuration fields changed");
    check(origins.size() == 4 && origins[1] == Coord{0, 0} &&
          origins[2] == Coord{4, 1} && origins[3] == Coord{7, 2}, "Origin indexing changed");
    auto event = InputHandler::readEvent(config.N, input);
    check(event.type == GameEventType::Move && event.player == 1, "Setup consumed part of the first event");
}  //----- end of testConfigurationAndOrigins


void testInvalidSetup()
// Algorithm : Reject incomplete, nonnumeric and out-of-range configurations before creating a game.
{
    for (const string text : {"", "4 4", "x 4 1 2 1", "0 4 1 2 1", "4 4 0 2 1",
                               "4 4 1 0 1", "4 4 1 2 0", "4 4 1 2 3", "1 1 1 2 1"})
    {
        istringstream input(text);
        bool rejected = false;

        try
        {
            InputHandler::readGameConfig(input);
        }
        catch (const invalid_argument &)
        {
            rejected = true;
        }

        check(rejected, "Invalid configuration was accepted: " + text);
    }

    istringstream incomplete("0 0\n1");
    bool rejected = false;

    try
    {
        InputHandler::readPlayersOrigins(2, incomplete);
    }
    catch (const invalid_argument &)
    {
        rejected = true;
    }

    check(rejected, "Incomplete origins were accepted");
}  //----- end of testInvalidSetup


void testTypedEvents()
// Algorithm : Decode every direction and verify zero-based event IDs are normalized exactly once.
{
    for (int action = 0; action < 4; action++)
    {
        istringstream input("move 1 " + ACTIONS[action] + "\r\n death 0\n");
        auto move = InputHandler::readEvent(3, input);
        auto death = InputHandler::readEvent(3, input);

        check(move.type == GameEventType::Move && move.player == 2 &&
              move.action == static_cast<Action>(action), "Move event was parsed incorrectly");
        check(death.type == GameEventType::Death && death.player == 1 && !death.action,
              "Death event was parsed incorrectly");
        check(InputHandler::readEvent(3, input).type == GameEventType::End, "EOF was not recognized");
    }
}  //----- end of testTypedEvents


void testMalformedEvents()
// Algorithm : Reject unknown, partial and oversized event lines without consuming the next valid event.
{
    for (const string text : {"move", "move 0", "move 0 diagonal", "death", "death -1",
                               "death 3", "death 99999999999999999999", "move x right",
                               "move 0 right extra", "death 0 extra", "unknown 0"})
    {
        istringstream input(text + "\nmove 0 right\n");
        auto invalid = InputHandler::readEvent(3, input);
        auto valid = InputHandler::readEvent(3, input);

        check(invalid.type == GameEventType::InputError && !invalid.error.empty(),
              "Malformed event was accepted: " + text);
        check(valid.type == GameEventType::Move && valid.player == 1, "Parser consumed the next event line");
    }
}  //----- end of testMalformedEvents


void testEndAndOutput()
// Algorithm : Distinguish EOF from stream failure and verify all typed actions produce exact protocol lines.
{
    istringstream empty(" \n\t");
    check(InputHandler::readEvent(2, empty).type == GameEventType::End, "Whitespace-only EOF failed");
    istringstream broken;
    broken.setstate(ios::badbit);
    check(InputHandler::readEvent(2, broken).type == GameEventType::InputError, "Stream failure became EOF");

    ostringstream output;

    for (int action = 0; action < 4; action++)
    {
        InputHandler::writeMove(static_cast<Action>(action), output);
    }

    check(output.str() == "left\nup\nright\ndown\n", "Output contains unexpected protocol text");
}  //----- end of testEndAndOutput


vector<TestCase> getTests()
// Algorithm : Register protocol parsing and output cases for independent execution.
{
    return {
        {"configuration_and_origins", testConfigurationAndOrigins},
        {"invalid_setup", testInvalidSetup},
        {"typed_events", testTypedEvents},
        {"malformed_events", testMalformedEvents},
        {"end_and_output", testEndAndOutput}
    };
}  //----- end of getTests
