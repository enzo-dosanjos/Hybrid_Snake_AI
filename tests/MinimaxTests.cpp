/*************************************************************************
MinimaxTests - Legal decisions, exhaustive outcomes and complete games
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//- Implementation of the module <MinimaxTests> (file MinimaxTests.cpp) --

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <algorithm>
#include <limits>

using namespace std;

//------------------------------------------------------- Personal Include
#include "GameTest.h"
#include "src/Agent/MinimaxAgent.h"


//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
int solve(GameEngine engine, int root, int remaining)
// Algorithm : Computes a reference win or loss by exhaustively exploring engine successors without heuristics or pruning.
{
    if (engine.isTerminalFor(root))
    {
        return engine.isAlive(root) ? 1 : -1;
    }

    check(remaining > 0, "Exhaustive reference exceeded finite-board turn bound");
    int player = engine.getState().currentPlayer;
    auto mask = engine.getActionMask(player);

    if (!engine.isAlive(player) || mask.none())
    {
        engine.playerDied(player);
        engine.finishTurn();
        return solve(engine, root, remaining - 1);
    }

    // Every opponent minimizes the same root player’s outcome, matching the multiplayer search contract.
    int result = player == root ? -1 : 1;

    for (int action = 0; action < 4; action++)
    {
        if (mask[action])
        {
            int outcome = solve(engine.successor(static_cast<Action>(action)), root, remaining - 1);
            result = player == root ? max(result, outcome) : min(result, outcome);
        }
    }

    return result;
}  //----- end of solve


void testOnlyLegalMove()
// Algorithm : Tests forced selection and state preservation using a position with exactly one legal move.
{
    MinimaxParams params;
    params.depth = 3;
    MinimaxAgent agent(params);
    GameEngine engine({4, 4, 1, 2, 1}, {{0, 0}, {0, 0}, {1, 0}});
    auto before = engine.getState();
    auto move = agent.selectAction(engine);

    check(move && *move == Action::Down, "Search ignored the only legal move");
    checkState(engine.getState(), before);
}  //----- end of testOnlyLegalMove


void testNoAction()
// Algorithm : Tests absent decisions by presenting trapped and already-finished positions to the agent.
{
    MinimaxAgent agent(MinimaxParams{});
    GameEngine trapped({2, 2, 1, 3, 1}, {{0, 0}, {0, 0}, {1, 0}, {0, 1}});
    check(!agent.selectAction(trapped), "Trapped player selected an action");

    GameEngine finished({3, 3, 1, 2, 1}, {{0, 0}, {0, 0}, {2, 2}});
    finished.playerDied(2);
    check(!agent.selectAction(finished), "Terminal game selected an action");
}  //----- end of testNoAction


void testExhaustiveSmallBoards()
// Algorithm : Tests optimal terminal outcomes by comparing selected moves with exhaustive search over thirty starting positions.
{
    MinimaxParams params;
    // Growth on every move bounds play on six cells; this depth includes forced deaths and turn advancement.
    params.depth = 14;
    MinimaxAgent agent(params);
    int positions = 0;

    for (int first = 0; first < 6; first++)
    {
        for (int second = 0; second < 6; second++)
        {
            if (first == second)
            {
                continue;
            }

            GameEngine engine({3, 2, 1, 2, 1},
                              {{0, 0}, {first % 3, first / 3}, {second % 3, second / 3}});
            auto before = engine.getState();
            auto move = agent.selectAction(engine);
            checkState(engine.getState(), before);
            check(move.has_value(), "Unblocked initial position returned no action");
            check(engine.getActionMask(1)[static_cast<int>(*move)], "Search returned illegal action");

            int best = solve(engine, 1, 14);
            int chosen = solve(engine.successor(*move), 1, 13);
            check(chosen == best, "Search chose a losing continuation when a win exists");
            positions++;
        }
    }

    check(positions == 30, "Not all ordered starting positions were checked");
}  //----- end of testExhaustiveSmallBoards


void testCompleteGames()
// Algorithm : Tests search integration by playing complete two- and three-player games while checking legality and state preservation.
{
    MinimaxParams params;
    params.depth = 3;
    MinimaxAgent agent(params);

    for (int players : {2, 3})
    {
        vector<Coord> origins{{0, 0}, {0, 0}, {3, 2}, {3, 0}};
        GameEngine engine({4, 3, 1, players, 1}, origins);
        int turns = 0;

        while (!engine.isTerminalState() && turns++ < 48)
        {
            auto before = engine.getState();
            int player = before.currentPlayer;

            if (!engine.isAlive(player))
            {
                engine.finishTurn();
                continue;
            }

            auto move = agent.selectAction(engine);
            checkState(engine.getState(), before);

            if (move)
            {
                check(engine.getActionMask(player)[static_cast<int>(*move)], "Illegal episode action");
                engine.updateStep(*move);
            }
            else
            {
                check(engine.getActionMask(player).none(), "Search refused legal moves");
                engine.playerDied(player);
                engine.finishTurn();
            }
        }

        check(engine.isTerminalState(), "Complete search-driven game did not terminate");
    }
}  //----- end of testCompleteGames


vector<TestCase> getTests()
// Algorithm : Registers this component’s named test functions for independent selection and execution.
{
    return {
        {"only_legal_move", testOnlyLegalMove},
        {"no_action", testNoAction},
        {"exhaustive_small_boards", testExhaustiveSmallBoards},
        {"complete_games", testCompleteGames}
    };
}  //----- end of getTests
