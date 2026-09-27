/*************************************************************************
ObservationTests - Board channels and state synchronization
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//- Implementation of the module <ObservationTests> (file ObservationTests.cpp) -

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <algorithm>

using namespace std;

//------------------------------------------------------- Personal Include
#include "GameTest.h"
#include "src/GameEngine/Observation.h"


//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
void checkObservation(const GameState &state, int perspective)
// Algorithm : Checks every encoded cell and channel against snake segments and coordinates from the supplied state.
{
    auto board = Observation{}.encode(state, perspective);
    int width = state.config.W;
    int height = state.config.H;
    Coord head = state.players.at(perspective).snake.front();

    check(board.size() == 9, "Wrong channel count");

    for (const auto &channel : board)
    {
        check(channel.size() == static_cast<size_t>(height), "Wrong height");

        for (const auto &row : channel)
        {
            check(row.size() == static_cast<size_t>(width), "Wrong width");
        }
    }

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            float expected[9] = {0};
            Coord cell{x, y};

            for (const auto &[id, player] : state.players)
            {
                int offset = id == perspective ? 0 : 3;

                for (size_t segment = 0; segment < player.snake.size(); segment++)
                {
                    if (player.snake[segment] == cell)
                    {
                        // A length-one snake is both head and tail; opponent presence is merged across players.
                        expected[offset] = max(expected[offset], segment == 0 ? 1.f : 0.f);
                        expected[offset + 2] = max(expected[offset + 2],
                                                  segment + 1 == player.snake.size() ? 1.f : 0.f);

                        if (segment > 0 && segment + 1 < player.snake.size())
                        {
                            expected[offset + 1] = 1.f;
                        }
                    }
                }
            }

            expected[6] = occupied(state, cell) ? 0.f : 1.f;
            expected[7] = width == 1 ? 0.f : head.x / float(width - 1);
            expected[8] = height == 1 ? 0.f : head.y / float(height - 1);

            for (int channel = 0; channel < 9; channel++)
            {
                checkNear(board[channel][y][x], expected[channel],
                          "Channel " + to_string(channel) + " at (" +
                          to_string(x) + ", " + to_string(y) + ")");
            }
        }
    }
}  //----- end of checkObservation


void testShapesAndPerspectives()
// Algorithm : Tests tensor dimensions and player perspectives on square, rectangular and one-cell-wide boards.
{
    for (auto size : vector<Coord>{{1, 5}, {5, 1}, {3, 7}, {8, 3}, {5, 5}})
    {
        GameEngine engine({size.x, size.y, 2, 2, 1},
                          {{0, 0}, {0, 0}, {size.x - 1, size.y - 1}});

        checkObservation(engine.getState(), 1);
        checkObservation(engine.getState(), 2);
    }
}  //----- end of testShapesAndPerspectives


void testBodyAndTailChannels()
// Algorithm : Tests segment channels by encoding snakes of different lengths from every player’s perspective.
{
    GameEngine engine({7, 5, 2, 3, 1}, {{0, 0}, {0, 0}, {6, 4}, {6, 0}});
    auto state = engine.getState();
    state.players.at(1).snake = {{2, 2}, {2, 1}, {1, 1}, {1, 2}};
    state.players.at(2).snake = {{4, 3}, {5, 3}};

    for (int player = 1; player <= 3; player++)
    {
        checkObservation(state, player);
    }
}  //----- end of testBodyAndTailChannels


void testGrowthAndTailRelease()
// Algorithm : Tests observation synchronization by comparing every cell after moves at several growth rates.
{
    for (int growth : {1, 2, 3})
    {
        GameEngine engine({12, 4, growth, 2, 1}, {{0, 0}, {0, 0}, {0, 3}});

        for (int step = 0; step < 16; step++)
        {
            engine.updateStep(Action::Right);
            checkObservation(engine.getState(), 1);
            checkObservation(engine.getState(), 2);
        }
    }
}  //----- end of testGrowthAndTailRelease


void testSnapshotEncoding()
// Algorithm : Tests stateless encoding by revisiting an old snapshot after encoding a newer position.
{
    GameEngine engine({6, 4, 2, 2, 1}, {{0, 0}, {0, 0}, {5, 3}});
    auto before = engine.getState();
    auto saved = before;
    Observation observation;
    auto first = observation.encode(before, 1);
    engine.updateStep(Action::Right);
    auto second = observation.encode(engine.getState(), 1);

    check(first != second, "Movement did not change encoded board");
    check(first == observation.encode(before, 1), "Encoder retained state from another snapshot");
    checkState(before, saved);
}  //----- end of testSnapshotEncoding


vector<TestCase> getTests()
// Algorithm : Registers this component’s named test functions for independent selection and execution.
{
    return {
        {"shapes_and_perspectives", testShapesAndPerspectives},
        {"body_and_tail_channels", testBodyAndTailChannels},
        {"growth_and_tail_release", testGrowthAndTailRelease},
        {"snapshot_encoding", testSnapshotEncoding}
    };
}  //----- end of getTests
