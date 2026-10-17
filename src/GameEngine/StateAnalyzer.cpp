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
vector<float> StateAnalyzer::computeExtraFeatures(const GameState &state, int perspective,
                                                 const FeatureSelection &selection) const
// Algorithm : Encode required state context, then append selected derived features in a fixed order.
// Required rules and timing :
//   board width, board height
//   growth interval, growth phase (round % M)
//   absolute round, scheduled player, perspective player
// Required player identities :
//   Own player first, then opponents in ascending ID order:
//     player ID, alive flag
// Optional spatial metrics (distances) :
//   For each opponent in ascending ID order:
//     own head → opponent segments: min, max, mean normalized distance
//     opponent head → own segments: min, max, mean normalized distance
// Optional size metric (snakeSize) :
//   own snake length / board area
// Optional game metric (gameProgress) :
//   capped elapsed scheduled slots / estimated turn budget
// Optional gameplay and aggressivity metrics :
//   For each opponent in ascending ID order:
//     legal-action count / 4 (mobility)
//     wall/corner proximity (wallProximity)
// Disabled groups are omitted; dead opponents retain zeroed optional metric slots.
{
    const auto &config = state.config;
    validateGameConfig(config);
    state.players.at(perspective);

    // Raw context avoids clipping distinct times or losing growth and scheduling information.
    vector<float> features = {
        float(config.W), float(config.H), float(config.M), float(state.round % config.M),
        float(state.round), float(state.currentPlayer), float(perspective)
    };
    vector<int> players{perspective};

    for (int player = 1; player <= config.N; player++)
    {
        if (player != perspective)
        {
            players.push_back(player);
        }
    }

    // IDs and alive flags follow the board's own-first slot ordering, including dead snakes.
    for (int player : players)
    {
        features.push_back(float(player));
        features.push_back(state.players.at(player).playerInfo.alive ? 1.f : 0.f);
    }

    if (selection.distances)
    {
        auto spatial = computeSpatialMetrics(state, perspective);
        features.insert(features.end(), spatial.begin(), spatial.end());
    }

    if (selection.snakeSize)
    {
        features.push_back(computeGrowthSizeMetrics(state, perspective)[2]);
    }

    if (selection.gameProgress)
    {
        features.push_back(computeGameProgress(state));
    }

    if (selection.mobility || selection.wallProximity)
    {
        auto gameplay = computeGameplayMetrics(state, perspective);

        for (size_t opponent = 0; opponent < gameplay.size(); opponent += 2)
        {
            if (selection.mobility)
            {
                features.push_back(gameplay[opponent]);
            }

            if (selection.wallProximity)
            {
                features.push_back(gameplay[opponent + 1]);
            }
        }
    }

    return features;
}  //----- end of computeExtraFeatures


vector<float> StateAnalyzer::computeSpatialMetrics(const GameState &state, int perspective) const
// Algorithm : Visit opponent IDs in order and retain six zero values for every dead opponent
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
            // Retain the six distance slots after an opponent dies
            features.insert(features.end(), 6, 0.f);
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
// Algorithm : Encode ordered opponent mobility and wall proximity without dependence on move history
{
    const auto &config = state.config;
    validateGameConfig(config);
    vector<float> features;

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
            features.insert(features.end(), 2, 0.f);
            continue;
        }

        if (opponent.snake.empty())
        {
            throw invalid_argument("Cannot analyze an empty snake");
        }

        // Mobility is the fraction of the four directions currently legal for this opponent
        features.push_back(opponent.playerInfo.actionMask.count() / 4.f);
        features.push_back(computeKillZoneProximity(opponent.snake.front(), config));
    }

    return features;
}  //----- end of computeGameplayMetrics


int StateAnalyzer::getFeatureCount(int numPlayers, const FeatureSelection &selection)
// Algorithm : Count required context and the enabled derived feature groups
{
    if (numPlayers < 1)
    {
        throw invalid_argument("Feature count requires at least one player");
    }

    int opponents = numPlayers - 1;
    return 7 + 2*numPlayers + 6*opponents*selection.distances + selection.snakeSize +
           selection.gameProgress + opponents*(selection.mobility + selection.wallProximity);
}  //----- end of getFeatureCount


//---------------------------------------------------------------- PRIVATE
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
