/*************************************************************************
SpaceRiskAnalyzer - Spatial evaluation of complete game snapshots
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//--- Interface of the module <SpaceRiskAnalyzer> (file SpaceRiskAnalyzer.h) ---
#ifndef SPACERISKANALYZER_H
#define SPACERISKANALYZER_H

//------------------------------------------------------------------------
// Role of the <SpaceRiskAnalyzer> module
// Estimates territory and movement risk with fixed snake occupancy.
// Results are heuristics; actual movement and growth belong to GameEngine.
// Territory uses shortest path lengths, with equal-distance cells unassigned.
// Risk compares scheduled arrivals, including the slots of dead players.
// Matrices use [y][x]; player vectors use IDs 1..N with slot zero unused.
//------------------------------------------------------------------------

//-------------------------------------------------------- Used interfaces
#include "../Structs.h"

//------------------------------------------------------------------ Types
struct ReachTiming {
    // Heads have arrival zero; other arrivals count scheduled slots from the snapshot.
    // Missing player entries mean no path within the requested move horizon.
    Matrix<std::unordered_map<int, long long>> slots;
    int numPlayers = 0;
};


struct AreaResult {
    std::vector<float> space;
    std::vector<float> blockRisk;
};


struct SpaceRiskResult {
    AreaResult areas;
    Matrix<float> heatmap;
};


//----------------------------------------------------------------- PUBLIC
class SpaceRiskAnalyzer {
//--------------------------------------------------------- Public Methods
    public:
        SpaceRiskResult analyzeState(const GameState &state, int player) const;
        // Contract : Return territory and perspective risk without modifying or retaining the snapshot.

        ReachTiming computeReachTiming(const GameState &state) const;
        // Contract : Return arrivals within maxLookahead moves per player; absent entries are unreachable.

        AreaResult calculateAreas(const GameState &state) const;
        // Contract : Count nearest-player territory including living heads; equal distances are contested and dead slots are zero.

        Matrix<float> calculateRiskHeatmap(const ReachTiming &timing, int player) const;
        // Contract : Return decayed opponent pressure on cells reachable by the selected player.

        float calculateDirectionalRisk(const Matrix<float> &heatmap, Coord head) const;
        // Contract : Sum risk in the in-bounds orthogonal neighbors of the head.

//---------------------------------------------- Constructors - destructor
        explicit SpaceRiskAnalyzer(int lookahead = 5, float decay = 0.7f);
        // Contract : Require nonnegative lookahead and finite decay in [0, 1].

//---------------------------------------------------------------- PRIVATE
    private:
        std::pair<Matrix<bool>, Matrix<int>> precomputeOccupancy(const GameState &state) const;
        // Contract : Validate geometry and mark all bodies occupied, with head IDs only for living players.

        ReachTiming computeReachTiming(const GameState &state, const Matrix<bool> &occupied, int limit) const;
        // Contract : Explore independently per living player; a negative limit explores all reachable cells.

        AreaResult calculateAreas(const GameState &state, const Matrix<bool> &occupied,
                                  const Matrix<int> &heads, const ReachTiming &timing) const;
        // Contract : Assign distance-based territory from full reach data and average blocking pressure per owner.

        void precomputeDecay();
        // Contract : Store one decay multiplier for each move distance up to maxLookahead.

        int maxLookahead;
        float decayFactor;
        std::vector<float> decayTable;
};

#endif //SPACERISKANALYZER_H
