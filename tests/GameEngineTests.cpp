/*************************************************************************
GameEngineTests - Game rules, scheduling and simulation
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//- Implementation of the module <GameEngineTests> (file GameEngineTests.cpp) -

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <random>
#include <set>

using namespace std;

//------------------------------------------------------- Personal Include
#include "GameTest.h"


//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
void testActionMasks()
// Algorithm : Tests legal moves against independent bounds and occupancy checks on varied board sizes.
{
    for (int width : {3, 5, 8})
    {
        for (int height : {2, 4, 7})
        {
            GameEngine engine({width, height, 2, 2, 1},
                              {{0, 0}, {0, 0}, {width - 1, height - 1}});
            GameState state = engine.getState();
            state.players.at(1).snake = {{1, 1}, {1, 0}, {0, 0}};
            GameEngine position(state);

            for (int player = 1; player <= 2; player++)
            {
                auto mask = position.getActionMask(player);
                Coord head = state.players.at(player).snake.front();

                for (int move = 0; move < 4; move++)
                {
                    const Coord directions[] = {{-1, 0}, {0, -1}, {1, 0}, {0, 1}};
                    Coord cell{head.x + directions[move].x, head.y + directions[move].y};
                    bool expected = cell.x >= 0 && cell.x < width &&
                                    cell.y >= 0 && cell.y < height && !occupied(state, cell);

                    check(mask[move] == expected, "Legal mask differs from board occupancy");
                }
            }
        }
    }
}  //----- end of testActionMasks


void testGrowthSchedule()
// Algorithm : Tests growth and turn timing by checking both snakes over twelve rounds at several growth rates.
{
    for (int growth : {1, 2, 3, 5})
    {
        GameEngine engine({30, 4, growth, 2, 1}, {{0, 0}, {0, 0}, {0, 3}});

        for (int round = 0; round < 12; round++)
        {
            for (int player = 1; player <= 2; player++)
            {
                const GameState before = engine.getState();
                engine.updateStep(Action::Right);
                const GameState after = engine.getState();
                const auto &snake = after.players.at(player).snake;
                size_t expected = before.players.at(player).snake.size() + (round % growth == 0);

                check(snake.size() == expected, "Growth occurred on the wrong round");
                check(snake.front() == Coord{round + 1, player == 1 ? 0 : 3}, "Wrong head");
                check(after.round == round + (player == 2), "Round advanced at the wrong slot");
                check(after.currentPlayer == (player == 1 ? 2 : 1), "Wrong next player");
            }
        }
    }
}  //----- end of testGrowthSchedule


void testCollisions()
// Algorithm : Tests elimination by moving each player into a wall, its own body and an opponent.
{
    for (int player : {1, 2})
    {
        GameEngine engine({5, 5, 2, 2, 1}, {{0, 0}, {0, 2}, {4, 2}});
        auto state = engine.getState();
        state.currentPlayer = player;
        GameEngine boundary(state);
        boundary.updateStep(player == 1 ? Action::Left : Action::Right);

        check(!boundary.isAlive(player), "Wall collision did not eliminate moving player");

        state.players.at(player).snake = {{2, 2}, {2, 1}, {1, 1}, {1, 2}};
        GameEngine self(state);
        self.updateStep(Action::Up);
        check(!self.isAlive(player), "Self collision did not eliminate moving player");

        state.players.at(player).snake = {{2, 2}};
        state.players.at(3 - player).snake = {{3, 2}, {3, 3}};
        GameEngine opponent(state);
        opponent.updateStep(Action::Right);
        check(!opponent.isAlive(player), "Opponent collision did not eliminate moving player");
    }
}  //----- end of testCollisions


void testDeathsAndClock()
// Algorithm : Tests death idempotency and terminal states while advancing through a dead player’s scheduled slot.
{
    GameEngine engine({6, 4, 2, 3, 1}, {{0, 0}, {0, 0}, {5, 3}, {5, 0}});
    engine.updateStep(Action::Down);
    engine.playerDied(2);
    const auto dead = engine.getState();
    engine.playerDied(2);
    checkState(engine.getState(), dead);
    check(!engine.isTerminalState(), "Two survivors must continue playing");
    check(engine.isTerminalFor(2), "Dead player's episode must end");
    check(engine.getState().currentPlayer == 2, "Death notification advanced turn");

    engine.finishTurn();
    engine.updateStep(Action::Down);

    check(engine.getState().round == 1 && engine.getState().currentPlayer == 1,
          "Dead slots must count towards rounds");

    engine.playerDied(3);
    check(engine.isTerminalState() && engine.isTerminalFor(1), "One survivor must end game");
}  //----- end of testDeathsAndClock


void testSuccessorIsolation()
// Algorithm : Tests independent search branches by comparing successors with applied moves and mutating one sibling.
{
    GameEngine engine({8, 4, 3, 2, 1}, {{0, 0}, {1, 1}, {6, 2}});
    const GameState original = engine.getState();
    auto first = engine.successor(Action::Right);
    auto second = engine.successor(Action::Down);
    GameEngine applied(original);
    applied.updateStep(Action::Right);

    checkState(first.getState(), applied.getState());
    checkState(engine.getState(), original);
    check(second.getState().players.at(1).snake.front() == Coord{1, 2}, "Sibling branch corrupted");

    first.playerDied(2);
    checkState(engine.getState(), original);
    check(second.isAlive(2), "Branch death leaked into sibling");
}  //----- end of testSuccessorIsolation


void testSeededGames()
// Algorithm : Tests game invariants by playing seeded legal games and checking bounds, overlap and termination.
{
    for (int seed = 0; seed < 12; seed++)
    {
        mt19937 gen(seed);
        GameEngine engine({7, 5, 1, 2, 1}, {{0, 0}, {0, 0}, {6, 4}});
        int turns = 0;

        while (!engine.isTerminalState() && turns++ < 80)
        {
            auto before = engine.getState();
            auto mask = engine.getActionMask(before.currentPlayer);
            vector<Action> moves;

            for (int action = 0; action < 4; action++)
            {
                if (mask[action])
                {
                    moves.push_back(static_cast<Action>(action));
                }
            }

            if (moves.empty())
            {
                engine.playerDied(before.currentPlayer);
                engine.finishTurn();
                continue;
            }

            uniform_int_distribution<int> choice(0, moves.size() - 1);
            auto next = engine.successor(moves[choice(gen)]);
            checkState(engine.getState(), before);
            engine = next;

            set<pair<int, int>> cells;

            for (const auto &[id, player] : engine.getState().players)
            {
                for (const Coord &cell : player.snake)
                {
                    check(cell.x >= 0 && cell.x < 7 && cell.y >= 0 && cell.y < 5,
                          "Legal play left the board");
                    check(cells.emplace(cell.x, cell.y).second, "Legal play created overlapping segments");
                }
            }
        }

        check(engine.isTerminalState(), "Growing snakes did not terminate on a finite board");
    }
}  //----- end of testSeededGames


void testTailAndDeadBodyCollisions()
// Algorithm : Tests retained occupancy by attempting entry into a moving tail and a dead opponent's body.
{
    GameEngine engine({5, 5, 3, 3, 1}, {{0, 0}, {2, 2}, {4, 4}, {4, 0}});
    GameState state = engine.getState();
    state.round = 1;
    state.players.at(1).snake = {{2, 2}, {2, 1}, {1, 1}, {1, 2}};
    GameEngine tail(state);
    tail.updateStep(Action::Left);

    check(!tail.isAlive(1), "Tail must remain occupied during collision checks");
    check(tail.getPlayers().at(1).snake == state.players.at(1).snake,
          "Collision must not alter the snake");

    state.players.at(1).snake = {{3, 4}};
    GameEngine corpse(state);
    corpse.playerDied(2);
    corpse.updateStep(Action::Right);
    check(!corpse.isAlive(1), "Dead bodies must remain obstacles");
}  //----- end of testTailAndDeadBodyCollisions


void testHistoryAndRejectedAction()
// Algorithm : Tests bounded move history and snapshot ownership after repeated moves and a rejected action.
{
    GameEngine engine({20, 4, 3, 2, 1}, {{0, 0}, {0, 0}, {0, 3}});

    for (int turn = 0; turn < 16; turn++)
    {
        engine.updateStep(Action::Right);
    }

    const GameState saved = engine.getState();

    for (int player = 1; player <= 2; player++)
    {
        check(saved.lastMoves.at(player) == deque<Action>(5, Action::Right),
              "History must retain only the five most recent moves");
    }

    bool rejected = false;

    try
    {
        engine.updateStep(static_cast<Action>(4));
    }
    catch (const invalid_argument &)
    {
        rejected = true;
    }

    check(rejected, "Invalid action must be rejected");
    checkState(engine.getState(), saved);

    GameEngine restored(saved);
    checkState(restored.getState(), saved);
    restored.updateStep(Action::Down);
    check(restored.getState().lastMoves.at(1).back() == Action::Down,
          "Latest action missing from restored history");
    checkState(engine.getState(), saved);
}  //----- end of testHistoryAndRejectedAction


vector<TestCase> getTests()
// Algorithm : Registers this component’s named test functions for independent selection and execution.
{
    return {
        {"action_masks", testActionMasks},
        {"growth_schedule", testGrowthSchedule},
        {"collisions", testCollisions},
        {"tail_and_dead_body_collisions", testTailAndDeadBodyCollisions},
        {"history_and_rejected_action", testHistoryAndRejectedAction},
        {"deaths_and_clock", testDeathsAndClock},
        {"successor_isolation", testSuccessorIsolation},
        {"seeded_games", testSeededGames}
    };
}  //----- end of getTests
