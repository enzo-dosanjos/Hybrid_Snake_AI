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
        std::vector<float> computeExtraFeatures(const GameState &state, int perspective) const;
        // Contract : Return schema version 1 features; opponent slots follow ascending IDs excluding perspective

        std::vector<float> computeSpatialMetrics(const GameState &state, int perspective) const;
        // Contract : Each opponent contributes alive, then min/max/average distances in both directions

        std::vector<float> computeGrowthSizeMetrics(const GameState &state, int perspective) const;
        // Contract : Return growth period / 10, fraction of rounds until growth and own occupied fraction

        float computeGameProgress(const GameState &state) const;
        // Contract : Return the original capped turn-budget heuristic, not a terminal-state test

        std::vector<float> computeGameplayMetrics(const GameState &state, int perspective) const;
        // Contract : Return own alive/entropy and scheduled ID, then opponent escape/wall/entropy triples

        static int getFeatureCount(int numPlayers);
        // Contract : Return 7 + 10*(N - 1) for a positive fixed player count

//---------------------------------------------------------------- PRIVATE
    private:
        float computeActionEntropy(const std::deque<Action> &moves) const;
        // Contract : Return Shannon entropy divided by two; empty and single-action histories return zero

        float computeKillZoneProximity(Coord head, const GameConfig &config) const;
        // Contract : Combine proximity to outer-quarter edges and the nearest wall, symmetrically

        static std::array<float, 3> distanceMetrics(Coord head, const std::deque<Coord> &body, float maxDistance);
        // Contract : Return normalized minimum, maximum and mean distances for a nonempty body and positive maxDistance
};

#endif //STATEANALYZER_H
