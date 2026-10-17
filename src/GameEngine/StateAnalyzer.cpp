/*************************************************************************
StateAnalyzer - Computes stable features from game snapshots
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//- Implementation of the module <StateAnalyzer> (file StateAnalyzer.cpp) -

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

using namespace std;

//------------------------------------------------------- Personal Include
#include "StateAnalyzer.h"
#include "../Utils.h"


//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
vector<float> StateAnalyzer::computeExtraFeatures(const GameState &state, int perspective) const
// Algorithm : Concatenate spatial, growth, progress and gameplay blocks in the documented schema order
// Spatial metrics :
//   For each opponent:
//     alive
//     own head → opponent segments: min, max, mean distance
//     opponent head → own segments: min, max, mean distance
// growth and size metrics :
//   growth interval
//   growth timing
//   own size
// game metrics :
//   game progress
//   own alive
//   own history entropy
//   scheduled player
// gameplay and aggressivity metrics :
//   For each opponent:
//     escape routes
//     kill-zone proximity
//     opponent history entropy
{
    validateGameConfig(state.config);
    vector<float> features = computeSpatialMetrics(state, perspective);
    vector<float> growth = computeGrowthSizeMetrics(state, perspective);
    vector<float> gameplay = computeGameplayMetrics(state, perspective);

    // Models interpret inputs by position, so these blocks must retain the schema order
    features.insert(features.end(), growth.begin(), growth.end());
    features.push_back(computeGameProgress(state));
    features.insert(features.end(), gameplay.begin(), gameplay.end());
    return features;
}  //----- end of computeExtraFeatures


vector<float> StateAnalyzer::computeSpatialMetrics(const GameState &state, int perspective) const
// Algorithm : Visit opponent IDs in order and retain seven zero values for every dead opponent
{
    const auto &own = state.players.at(perspective);
    const auto &config = state.config;
    validateGameConfig(config);

    if (own.snake.empty())
    {
        throw invalid_argument("Cannot analyze an empty snake");
    }

    vector<float> features;
    // Use the board diagonal as a distance bound, including for single-cell dimensions
    float maxDistance = hypot(float(config.W), float(config.H));

    // Iterate by ID rather than container order to keep each opponent in a stable slot
    for (int player = 1; player <= config.N; player++)
    {
        if (player == perspective)
        {
            continue;
        }

        const auto &opponent = state.players.at(player);

        if (!opponent.playerInfo.alive)
        {
            // Retain the alive flag and six distance slots after an opponent dies
            features.insert(features.end(), 7, 0.f);
            continue;
        }

        if (opponent.snake.empty())
        {
            throw invalid_argument("Cannot analyze an empty snake");
        }

        // Measure both directions because the heads and snake shapes differ
        // Each target snake includes all segments, including its head and tail
        auto toOpponent = distanceMetrics(own.snake.front(), opponent.snake, maxDistance);
        auto toOwn = distanceMetrics(opponent.snake.front(), own.snake, maxDistance);
        features.push_back(1.f);
        features.insert(features.end(), toOpponent.begin(), toOpponent.end());
        features.insert(features.end(), toOwn.begin(), toOwn.end());
    }

    return features;
}  //----- end of computeSpatialMetrics


vector<float> StateAnalyzer::computeGrowthSizeMetrics(const GameState &state, int perspective) const
// Algorithm : Use the authoritative round for growth phase and normalize length by board area
{
    const auto &config = state.config;
    validateGameConfig(config);
    // The final modulo maps a growth round to zero instead of a full remaining period
    int roundsUntilGrowth = (config.M - state.round % config.M) % config.M;

    // Larger M means slower growth; division by 10 retains the existing feature scale
    // The remaining values express the fraction of a growth period and occupied board area
    return {
        config.M / 10.f,  // max value for M by game definition
        roundsUntilGrowth / float(config.M),
        state.players.at(perspective).snake.size() / (float(config.W) * config.H)
    };
}  //----- end of computeGrowthSizeMetrics


float StateAnalyzer::computeGameProgress(const GameState &state) const
// Algorithm : Normalize elapsed scheduled slots using the existing board-area and growth-period heuristic
{
    const auto &config = state.config;
    validateGameConfig(config);
    // This is an estimated duration for scaling the feature, not an enforced game limit
    float maxTurns = float(config.W) * config.H * config.M;

    // Preserve the original shorter duration estimate when growth is not every round
    if (config.M > 1)
    {
        maxTurns *= 0.8f;
    }

    // Original player count includes dead slots; deaths must not accelerate the clock
    float turn = float(state.round) * config.N + state.currentPlayer - 1;
    return clamp(turn / maxTurns, 0.f, 1.f);
}  //----- end of computeGameProgress


vector<float> StateAnalyzer::computeGameplayMetrics(const GameState &state, int perspective) const
// Algorithm : Encode own status and turn context, then ordered opponent mobility, wall proximity and history
{
    const auto &config = state.config;
    validateGameConfig(config);
    const auto &own = state.players.at(perspective);
    auto history = state.lastMoves.find(perspective);
    float entropy = 0.f;

    // Dead players and missing histories contribute zero movement entropy
    // Entropy is already in [0, 1]; tanh compresses it to [0, tanh(1)]
    if (own.playerInfo.alive && history != state.lastMoves.end())
    {
        entropy = tanh(computeActionEntropy(history->second));
    }

    // Encode the absolute scheduled ID in [0, 1], avoiding division by zero for N = 1
    vector<float> features = {
        own.playerInfo.alive ? 1.f : 0.f,
        entropy,
        config.N == 1 ? 0.f : (state.currentPlayer - 1) / float(config.N - 1)
    };

    for (int player = 1; player <= config.N; player++)
    {
        if (player == perspective)
        {
            continue;
        }

        const auto &opponent = state.players.at(player);

        if (!opponent.playerInfo.alive)
        {
            // Keep the same opponent slots as the spatial block, even after death
            features.insert(features.end(), 3, 0.f);
            continue;
        }

        if (opponent.snake.empty())
        {
            throw invalid_argument("Cannot analyze an empty snake");
        }

        // Use this opponent's history rather than reusing the perspective player's moves
        history = state.lastMoves.find(player);
        entropy = history == state.lastMoves.end() ? 0.f : tanh(computeActionEntropy(history->second));
        // Mobility is the fraction of the four directions currently legal for this opponent
        features.push_back(opponent.playerInfo.actionMask.count() / 4.f);
        features.push_back(computeKillZoneProximity(opponent.snake.front(), config));
        features.push_back(entropy);
    }

    return features;
}  //----- end of computeGameplayMetrics


int StateAnalyzer::getFeatureCount(int numPlayers)
// Algorithm : Add seven global values to the ten retained values per opponent
{
    if (numPlayers < 1)
    {
        throw invalid_argument("Feature count requires at least one player");
    }

    // Seven shared values plus seven spatial and three gameplay values per opponent
    return 7 + 10 * (numPlayers - 1);
}  //----- end of getFeatureCount


//---------------------------------------------------------------- PRIVATE
float StateAnalyzer::computeActionEntropy(const deque<Action> &moves) const
// Algorithm : Count the four actions and normalize Shannon entropy by its two-bit maximum
{
    if (moves.empty())
    {
        return 0.f;
    }

    // Only action frequencies matter here; entropy does not preserve movement order
    array<int, 4> counts{};

    for (Action move : moves)
    {
        counts.at(static_cast<size_t>(move))++;
    }

    float entropy = 0.f;

    for (int count : counts)
    {
        // An absent action contributes zero; skip it to avoid evaluating log2(0)
        if (count > 0)
        {
            float probability = count / float(moves.size());
            entropy -= probability * log2(probability);
        }
    }

    // Four equally frequent actions give log2(4) = 2 bits; one repeated action gives zero
    return entropy / 2.f;
}  //----- end of computeActionEntropy


float StateAnalyzer::computeKillZoneProximity(Coord head, const GameConfig &config) const
// Algorithm : Average symmetric outer-quarter edge weights with normalized nearest-wall proximity
{
    validateGameConfig(config);
    // Distances to the nearest wall on each axis make opposite edges equivalent
    float horizontal = min(head.x, config.W - 1 - head.x);
    float vertical = min(head.y, config.H - 1 - head.y);
    float cornerWeight = 0.f;

    // Each axis contributes half a point in its outer quarter, emphasizing corners
    if (horizontal < config.W / 4.f)
    {
        cornerWeight += 0.5f;
    }

    if (vertical < config.H / 4.f)
    {
        cornerWeight += 0.5f;
    }

    // Combine the corner indicator with continuous wall proximity using the longer side
    // This geometric heuristic does not establish that an opponent is actually trapped
    float wall = 1.f - min(horizontal, vertical) / (max(config.W, config.H) / 2.f);
    return clamp((cornerWeight + wall) / 2.f, 0.f, 1.f);
}  //----- end of computeKillZoneProximity


array<float, 3> StateAnalyzer::distanceMetrics(Coord head, const deque<Coord> &body, float maxDistance)
// Algorithm : Accumulate normalized Euclidean min, max and mean distances from a head to all segments
{
    if (body.empty())
    {
        throw invalid_argument("Cannot measure distances to an empty snake");
    }

    // Slots hold minimum, maximum and mean; normalized distances are bounded by one
    array<float, 3> result{1.f, 0.f, 0.f};

    for (const Coord &cell : body)
    {
        float distance = hypot(float(head.x) - cell.x, float(head.y) - cell.y) / maxDistance;
        result[0] = min(result[0], distance);
        result[1] = max(result[1], distance);
        // Accumulate each segment's share of the mean, giving every segment equal weight
        result[2] += distance / body.size();
    }

    return result;
}  //----- end of distanceMetrics
