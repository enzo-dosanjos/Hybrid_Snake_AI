/*************************************************************************
SpaceRiskAnalyzer - A spatial risk assessment system for grid-based games
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

#ifndef SPACERISKANALYZER_H
#define SPACERISKANALYZER_H

//------------------------------------------------------------------------
// Role of the <SpaceRiskAnalyzer> module
// The SpaceRiskAnalyzer module provides tools for analyzing spatial risks
// in a grid-based game environment. It computes risk heatmaps showing
// dangerous areas, calculates accessible space for each player, and
// evaluates movement risks. The module helps the minimax algorithm to
// make informed decisions.
//------------------------------------------------------------------------

//-------------------------------------------------------- Used interfaces
#include "GameEngine.h"
#include "Reward.h"
#include "Structs.h"

//----------------------------------------------------------------- PUBLIC
class SpaceRiskAnalyzer
{
//--------------------------------------------------------- Public Methods
    public:
        std::pair<std::pair<std::vector<float>, std::vector<float>>, Matrix<float>> analyzeState(
            const Players &playerState,
            int currentPlayer
        );
        // Usage :
        //
        // Contract :
        //

        Matrix<float> calculateRiskHeatmap(
            const int currentPlayer
        );
        // Usage :
        //
        // Contract :
        //

        std::pair<std::vector<float>, std::vector<float>> calculateAreas(
            const Players &playerState,
            const Matrix<bool> &occupied,
            const Matrix<int> &heads
        );
        // Usage :
        //
        // Contract :
        //

        float calculateDirectionalRisk(
            const Matrix<float> &heatmap,
            Coord head
        );
        // Usage :
        //
        // Contract :
        //

//---------------------------------------------- Constructors - destructor
    SpaceRiskAnalyzer(
            const GameConfig &p_config,
            int lookahead = 5,
            float decay = 0.7
    ) : config(p_config),
        maxLookahead(lookahead),
        decayFactor(decay)
    {
        precomputeDecay();
    }

//-------------------------------------------------------------- PROTECTED
    protected:
        Matrix<std::unordered_map<int, int>> computeReachTiming(
            const Players &playerState,
            const Matrix<bool> &occupied
        );
        // Usage :
        //
        // Contract :
        //

        Matrix<int> precomputeCurrentPlayerTiming(
            const Matrix<std::unordered_map<int, int>> &timingGrid,
            int currentPlayer
        );
        // Usage :
        //
        // Contract :
        //

        std::pair<Matrix<bool>, Matrix<int>> precomputeOccupancy(
            const Players &playerState
        );
        // Usage :
        //
        // Contract :
        //

        void precomputeDecay();
        // Usage :
        //
        // Contract :
        //

//---------------------------------------------------------------- PRIVATE
    private:
        GameConfig config;
        int maxLookahead;
        float decayFactor;
        std::vector<float> decayTable;

        // keep previous data to avoid recomputing the same data
        Matrix<std::unordered_map<int, int>> lastReachData;
        Players lastPlayerState;
        Matrix<bool> lastOccupied;
        Matrix<int> lastHeads;
};


#endif //SPACERISKANALYZER_H
