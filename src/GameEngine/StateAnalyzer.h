/*************************************************************************
StateAnalyzer - Computes stable features from game snapshots
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//---- Interface of the module <StateAnalyzer> (file StateAnalyzer.h) ----
#ifndef STATEANALYZER_H
#define STATEANALYZER_H

//------------------------------------------------------------------------
// Role of the <StateAnalyzer> module
// Computes versioned numerical features with explicit player perspective
// Feature order and normalization are documented in README.md (Game State
// Representation)
//------------------------------------------------------------------------

//-------------------------------------------------------- Used interfaces
#include <array>

#include "../Structs.h"

//----------------------------------------------------------------- PUBLIC
class StateAnalyzer {
//--------------------------------------------------------- Public Methods
    public:
        std::vector<float> computeExtraFeatures(const GameState &state, int perspective,
                                                const FeatureSelection &selection = {}) const;
        // Contract : Return schema version 2 features; opponent slots follow ascending IDs excluding perspective

        std::vector<float> computeSpatialMetrics(const GameState &state, int perspective) const;
        // Contract : Each opponent contributes min/max/average distances in both directions

        std::vector<float> computeGrowthSizeMetrics(const GameState &state, int perspective) const;
        // Contract : Return growth period / 10, fraction of rounds until growth and own occupied fraction

        float computeGameProgress(const GameState &state) const;
        // Contract : Return the original capped turn-budget heuristic, not a terminal-state test

        std::vector<float> computeGameplayMetrics(const GameState &state, int perspective) const;
        // Contract : Return opponent escape and wall-proximity pairs in ascending ID order

        static int getFeatureCount(int numPlayers, const FeatureSelection &selection = {});
        // Contract : Return required context plus the selected derived features for a fixed player count

//---------------------------------------------------------------- PRIVATE
    private:
        float computeKillZoneProximity(Coord head, const GameConfig &config) const;
        // Contract : Combine proximity to outer-quarter edges and the nearest wall, symmetrically

        static std::array<float, 3> distanceMetrics(Coord head, const std::deque<Coord> &body, float maxDistance);
        // Contract : Return normalized minimum, maximum and mean distances for a nonempty body and positive maxDistance
};

#endif //STATEANALYZER_H
