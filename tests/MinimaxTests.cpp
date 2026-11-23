/*************************************************************************
MinimaxTests - Legal decisions, exhaustive outcomes and complete games
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//- Implementation of the module <MinimaxTests> (file MinimaxTests.cpp) --

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <algorithm>
#include <cmath>
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


float referenceScore(GameEngine engine, int root, int remaining, const Rewards &weights)
// Algorithm : Compute the fixed-depth value by enumerating every legal branch and evaluating each leaf independently.
{
    if (engine.isTerminalFor(root))
    {
        // Use infinity as a reference marker so the test need not share the search's terminal score constant.
        float terminal = numeric_limits<float>::infinity();
        return engine.isAlive(root) ? terminal : -terminal;
    }

    if (remaining == 0)
    {
        SpaceRiskAnalyzer analyzer;
        double ownScore = weights.snakeLength * engine.getPlayers().at(root).snake.size();
        double opponentScore = 0;

        // Analyze each living perspective separately using the analyzer's complete snapshot result.
        for (const auto &[player, data] : engine.getPlayers())
        {
            if (!data.playerInfo.alive)
            {
                continue;
            }

            auto [areas, heatmap] = analyzer.analyzeState(engine.getState(), player);
            float heat = analyzer.calculateDirectionalRisk(heatmap, data.snake.front());

            if (player == root)
            {
                ownScore += weights.selfSpace * areas.space[player] + weights.selfBlockRisk * areas.blockRisk[player]
                          + weights.selfHeatRisk * heat;
            }
            else
            {
                opponentScore += weights.oppSpace * areas.space[player] + weights.oppBlockRisk * areas.blockRisk[player]
                               + weights.oppHeatRisk * heat;
            }
        }

        return ownScore - opponentScore;
    }

    int player = engine.getState().currentPlayer;
    ActionMask mask = engine.getActionMask(player);

    if (mask.none())
    {
        if (engine.isAlive(player))
        {
            engine.playerDied(player);
        }

        engine.finishTurn();
        return referenceScore(engine, root, remaining - 1, weights);
    }

    vector<float> scores;

    for (size_t action = 0; action < mask.size(); action++)
    {
        if (mask[action])
        {
            scores.push_back(referenceScore(engine.successor(static_cast<Action>(action)), root, remaining - 1, weights));
        }
    }

    return player == root ? *max_element(scores.begin(), scores.end()) : *min_element(scores.begin(), scores.end());
}  //----- end of referenceScore


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

    GameEngine deadSlot({3, 3, 1, 3, 1}, {{0, 0}, {0, 0}, {2, 2}, {0, 2}});
    deadSlot.playerDied(1);
    auto before = deadSlot.getState();
    check(!agent.selectAction(deadSlot), "Dead scheduled player selected an action");
    checkState(deadSlot.getState(), before);
}  //----- end of testNoAction


void testExactMoveScores()
// Algorithm : Compare every legal score with unpruned search across a growth boundary and a dead player's slot.
{
    GameEngine engine({5, 4, 2, 4, 1}, {{0, 0}, {0, 0}, {2, 0}, {4, 0}, {2, 2}});
    engine.updateStep(Action::Down);
    engine.playerDied(2);
    engine.finishTurn();
    engine.updateStep(Action::Down);
    auto before = engine.getState();
    int root = before.currentPlayer;
    ActionMask mask = engine.getActionMask(root);

    for (int depth : {1, 3, 5, 6})
    {
        MinimaxParams params;
        params.depth = depth;
        Minimax search(params);
        auto scores = search.calculateMoveScores(engine);
        checkState(engine.getState(), before);
        check(scores.size() == mask.count(), "Search must score exactly the legal actions");

        for (const auto &[action, score] : scores)
        {
            check(mask[static_cast<size_t>(action)], "Search scored an illegal action");
            float expected = referenceScore(engine.successor(action), root, depth - 1, params.rewards);

            if (isinf(expected))
            {
                check(isfinite(score) && copysign(1.f, score) == copysign(1.f, expected),
                      "Terminal action must have a finite score with the correct sign");
            }
            else
            {
                checkNear(score, expected, "Inexact root action score at depth " + to_string(depth)
                                           + ", action " + to_string(static_cast<int>(action)));
            }
        }
    }
}  //----- end of testExactMoveScores


void testTerminalDominance()
// Algorithm : Compare forced wins and losses with continuing branches under extreme heuristic weights.
{
    GameEngine engine({3, 2, 1, 3, 1}, {{0, 0}, {0, 0}, {2, 0}, {2, 1}});
    engine.playerDied(3);
    auto before = engine.getState();
    MinimaxParams params;
    params.depth = 2;
    params.rewards.snakeLength = numeric_limits<double>::max();
    auto scores = Minimax(params).calculateMoveScores(engine);
    float win = scores.at(Action::Right);
    float continuing = scores.at(Action::Down);

    check(isfinite(win) && isfinite(continuing) && win > continuing && continuing > 0.f,
          "Forced win must dominate even a very large positive heuristic");
    MinimaxAgent agent(params);
    Agent &decisionAgent = agent;
    check(decisionAgent.selectAction(engine) == Action::Right, "Agent ignored the guaranteed win");
    checkState(engine.getState(), before);

    GameEngine losing({4, 2, 1, 3, 1}, {{0, 0}, {1, 0}, {3, 1}, {3, 0}});
    losing.playerDied(3);
    auto beforeLoss = losing.getState();
    params.depth = 4;
    params.rewards.snakeLength = -numeric_limits<double>::max();
    scores = Minimax(params).calculateMoveScores(losing);
    float loss = scores.at(Action::Right);
    continuing = scores.at(Action::Left);

    check(isfinite(loss) && isfinite(continuing) && loss < continuing && continuing < 0.f,
          "Forced loss must score below even a very large negative heuristic");
    checkState(losing.getState(), beforeLoss);
}  //----- end of testTerminalDominance


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
        {"exact_move_scores", testExactMoveScores},
        {"terminal_dominance", testTerminalDominance},
        {"exhaustive_small_boards", testExhaustiveSmallBoards},
        {"complete_games", testCompleteGames}
    };
}  //----- end of getTests
