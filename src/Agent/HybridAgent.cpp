/*************************************************************************
HybridAgent - Combines Minimax and learned action scores
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//-- Implementation of the module <HybridAgent> (file HybridAgent.cpp) --

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>

using namespace std;

//------------------------------------------------------- Personal Include
#include "HybridAgent.h"
#include "../GameEngine/DecisionEncoding.h"

//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
optional<Action> HybridAgent::selectAction(const GameEngine &engine)
// Algorithm : Explore through the learner or maximize a weighted sum of rescaled legal scores.
{
    int player = engine.getState().currentPlayer;
    ActionMask mask = engine.getActionMask(player);
    if (engine.isTerminalFor(player) || mask.none())
    {
        return nullopt;
    }

    if (explore)
    {
        auto action = dqn.sampleExplorationAction(mask);
        if (action)
        {
            return action;
        }
    }

    // Use the original ranking at the endpoints, where only one scorer contributes.
    if (!margin && lambda == 1.f)
    {
        return dqn.selectAction(encodeDecision(engine, player, dqn.getFeatureSelection()), false);
    }

    auto searchScores = minimax.calculateMoveScores(engine);
    if (margin)
    {
        auto best = max_element(searchScores.begin(), searchScores.end(), [](const auto &left, const auto &right)
        {
            return left.second < right.second;
        });
        float terminal = numeric_limits<float>::max() / 4.f;
        if (abs(best->second) == terminal || searchScores.size() == 1)
        {
            return best->first;
        }

        auto values = dqn.forwardPropagation(encodeDecision(engine, player, dqn.getFeatureSelection()));
        Action choice = best->first;
        for (const auto &[action, score] : searchScores)
        {
            if (score != -terminal && double(best->second) - score <= *margin &&
                values[static_cast<int>(action)] > values[static_cast<int>(choice)])
            {
                choice = action;
            }
        }
        return choice;
    }

    if (lambda == 0.f)
    {
        return max_element(searchScores.begin(), searchScores.end(), [](const auto &left, const auto &right)
        {
            return left.second < right.second;
        })->first;
    }

    array<float, 4> searchValues{};
    for (const auto &[action, score] : searchScores)
    {
        searchValues[static_cast<size_t>(action)] = score;
    }

    auto qValues = dqn.forwardPropagation(encodeDecision(engine, player, dqn.getFeatureSelection()));
    auto normalizedSearch = normalizeLegalScores(searchValues, mask);
    auto normalizedQ = normalizeLegalScores(qValues, mask);

    optional<Action> best;
    double bestScore = 0.0;
    for (size_t action = 0; action < mask.size(); action++)
    {
        if (!mask[action])
        {
            continue;
        }

        double combined = (1.0 - lambda) * normalizedSearch[action] + lambda * normalizedQ[action];
        if (!best || combined > bestScore)
        {
            best = static_cast<Action>(action);
            bestScore = combined;
        }
    }

    return best;
}  //----- end of selectAction


HybridAgent::HybridAgent(DQN &p_dqn, const MinimaxParams &params, float p_lambda, bool p_explore,
                         optional<float> p_margin)
    : dqn(p_dqn), minimax(params), lambda(p_lambda), explore(p_explore), margin(p_margin)
// Algorithm : Keep the shared learner, configure search and validate the neural weighting.
{
    if (margin && (!isfinite(*margin) || *margin < 0.f || explore))
    {
        throw invalid_argument("Override margin requires greedy play and a finite nonnegative score gap");
    }
    if (!(lambda >= 0.f && lambda <= 1.f))
    {
        throw invalid_argument("HybridAgent lambda must be in [0, 1]");
    }
}  //----- end of HybridAgent


//---------------------------------------------------------------- PRIVATE
array<double, 4> HybridAgent::normalizeLegalScores(const array<float, 4> &scores, const ActionMask &mask)
// Algorithm : Map the legal minimum and maximum to zero and one, leaving equal scores at zero.
{
    float minimum = numeric_limits<float>::infinity();
    float maximum = -minimum;

    for (size_t action = 0; action < mask.size(); action++)
    {
        if (mask[action])
        {
            minimum = min(minimum, scores[action]);
            maximum = max(maximum, scores[action]);
        }
    }

    array<double, 4> normalized{};
    if (minimum == maximum)
    {
        return normalized;
    }

    // Subtract in double so the full range of finite float scores fits.
    double range = static_cast<double>(maximum) - minimum;
    for (size_t action = 0; action < mask.size(); action++)
    {
        if (mask[action])
        {
            normalized[action] = (static_cast<double>(scores[action]) - minimum) / range;
        }
    }

    return normalized;
}  //----- end of normalizeLegalScores
