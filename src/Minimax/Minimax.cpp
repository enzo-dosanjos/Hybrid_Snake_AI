/*************************************************************************
Minimax - Scores legal moves with alpha-beta search
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//------ Implementation of the module <Minimax> (file Minimax.cpp) ------

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

using namespace std;

//------------------------------------------------------- Personal Include
#include "Minimax.h"

//-------------------------------------------------------------- Constants
namespace {
    // Leave room for the heuristic range and for subtracting any two scores.
    constexpr float TERMINAL_SCORE = numeric_limits<float>::max() / 4.f;
    constexpr long double HEURISTIC_LIMIT = TERMINAL_SCORE / 2.f;
    constexpr float INFINITY_SCORE = numeric_limits<float>::infinity();
}

//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
map<Action, float> Minimax::calculateMoveScores(const GameEngine &engine) const
// Algorithm : Score every legal engine successor with a fresh full window so each result is exact.
{
    int rootPlayer = engine.getState().currentPlayer;
    map<Action, float> scores;

    if (engine.isTerminalFor(rootPlayer))
    {
        return scores;
    }

    ActionMask mask = engine.getActionMask(rootPlayer);

    for (size_t action = 0; action < mask.size(); action++)
    {
        if (mask[action])
        {
            Action move = static_cast<Action>(action);
            GameEngine next = engine.successor(move);
            scores.emplace(move, minimax(next, params.depth - 1, -INFINITY_SCORE, INFINITY_SCORE, rootPlayer));
        }
    }

    return scores;
}  //----- end of calculateMoveScores


Minimax::Minimax(const MinimaxParams &p_params, int lookahead, float decay)
    : params(p_params), analyzer(lookahead, decay)
// Algorithm : Initialize the spatial analyzer and validate the search horizon and reward weights.
{
    if (params.depth < 1)
    {
        throw invalid_argument("Minimax depth must be positive");
    }

    const Rewards &weights = params.rewards;

    for (double weight : {weights.selfSpace, weights.selfBlockRisk, weights.selfHeatRisk, weights.snakeLength,
                          weights.oppSpace, weights.oppBlockRisk, weights.oppHeatRisk})
    {
        if (!isfinite(weight))
        {
            throw invalid_argument("Minimax reward weights must be finite");
        }
    }
}  //----- end of Minimax


//---------------------------------------------------------------- PRIVATE
float Minimax::evaluateState(const GameState &state, int rootPlayer) const
// Algorithm : Weight the root's territory, risks and length, then subtract each living opponent's weighted territory and risks.
{
    const Rewards &weights = params.rewards;
    AreaResult areas = analyzer.calculateAreas(state);
    ReachTiming timing = analyzer.computeReachTiming(state);
    const auto &rootSnake = state.players.at(rootPlayer).snake;
    float rootHeatRisk = analyzer.calculateDirectionalRisk(analyzer.calculateRiskHeatmap(timing, rootPlayer),
                                                           rootSnake.front());

    // Accumulate in a wider type before clamping so large finite weights cannot overlap terminal scores.
    long double score = areas.space[rootPlayer] * static_cast<long double>(weights.selfSpace)
                      + areas.blockRisk[rootPlayer] * static_cast<long double>(weights.selfBlockRisk)
                      + rootHeatRisk * static_cast<long double>(weights.selfHeatRisk)
                      + rootSnake.size() * static_cast<long double>(weights.snakeLength);

    for (const auto &[opponent, data] : state.players)
    {
        if (opponent == rootPlayer || !data.playerInfo.alive)
        {
            continue;
        }

        float opponentHeatRisk = analyzer.calculateDirectionalRisk(analyzer.calculateRiskHeatmap(timing, opponent),
                                                                   data.snake.front());
        score -= areas.space[opponent] * static_cast<long double>(weights.oppSpace)
               + areas.blockRisk[opponent] * static_cast<long double>(weights.oppBlockRisk)
               + opponentHeatRisk * static_cast<long double>(weights.oppHeatRisk);
    }

    return static_cast<float>(clamp(score, -HEURISTIC_LIMIT, HEURISTIC_LIMIT));
}  //----- end of evaluateState


float Minimax::minimax(const GameEngine &engine, int depth, float alpha, float beta, int rootPlayer) const
// Algorithm : Resolve terminal outcomes, then search scheduled slots with the root maximizing and opponents minimizing.
{
    if (engine.isTerminalFor(rootPlayer))
    {
        return engine.isAlive(rootPlayer) ? TERMINAL_SCORE : -TERMINAL_SCORE;
    }

    if (depth == 0)
    {
        return evaluateState(engine.getState(), rootPlayer);
    }

    int currentPlayer = engine.getState().currentPlayer;

    if (!engine.isAlive(currentPlayer))
    {
        GameEngine next = engine;
        next.finishTurn();
        return minimax(next, depth - 1, alpha, beta, rootPlayer);
    }

    ActionMask mask = engine.getActionMask(currentPlayer);

    if (mask.none())
    {
        // The engine treats any direction as a collision when none is legal.
        GameEngine next = engine.successor(Action::Left);
        return minimax(next, depth - 1, alpha, beta, rootPlayer);
    }

    bool maximizing = currentPlayer == rootPlayer;
    float bestScore = maximizing ? -INFINITY_SCORE : INFINITY_SCORE;

    for (size_t action = 0; action < mask.size(); action++)
    {
        if (!mask[action])
        {
            continue;
        }

        GameEngine next = engine.successor(static_cast<Action>(action));
        float score = minimax(next, depth - 1, alpha, beta, rootPlayer);

        if (maximizing)
        {
            bestScore = max(bestScore, score);
            alpha = max(alpha, bestScore);
        }
        else
        {
            bestScore = min(bestScore, score);
            beta = min(beta, bestScore);
        }

        if (alpha >= beta)
        {
            break;
        }
    }

    return bestScore;
}  //----- end of minimax
