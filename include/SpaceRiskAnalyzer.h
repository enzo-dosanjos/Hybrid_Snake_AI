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
#include "Utils.h"
#include "Structs.h"

//----------------------------------------------------------------- PUBLIC
class SpaceRiskAnalyzer
{
//--------------------------------------------------------- Public Methods
    public:
        std::pair<std::pair<std::vector<float>, std::vector<float>>, Matrix<float>> analyzeState(
            const PlayersData &playerState,
            int currentPlayer
        );
        // Usage :
        //
        // Contract :
        //

        Matrix<float> calculateRiskHeatmap(
            const PlayersData &playerState,
            const Matrix<bool> &occupied,
            int currentPlayer
        );
        // Usage :
        //
        // Contract :
        //

        std::pair<std::vector<float>, std::vector<float>> calculateAreas(
            const PlayersData &playerState,
            const Matrix<bool> &occupied,
            const Matrix<int> &heads
        );
        // Usage :
        //
        // Contract :
        //

        float calculateDirectionalRisk(
            const std::vector<std::vector<float>> &heatmap,
            Coord head
        );
        // Usage :
        //
        // Contract :
        //

//---------------------------------------------- Constructors - destructor
    SpaceRiskAnalyzer(
            int w,
            int h,
            int m,
            int n,
            int p,
            int boardC,
            int lookahead = 5,
            float decay = 0.7
    ) : W(w),
        H(h),
        maxLookahead(lookahead),
        decayFactor(decay),
        helper(w, h, m, n, p, boardC)
    {
        precomputeDecay();
    }

//-------------------------------------------------------------- PROTECTED
    protected:
        Matrix<std::unordered_map<int, int>> computeReachTiming(
            const PlayersData &playerState,
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
            const PlayersData &playerState
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
        int W, H;
        int maxLookahead;
        float decayFactor;
        std::vector<float> decayTable;

        // keep previous data to avoid recomputing the same data
        Matrix<std::unordered_map<int, int>> lastReachData;
        PlayersData lastPlayerState;
        Matrix<bool> lastOccupied;
        Matrix<int> lastHeads;

        Utils helper;
};


#endif //SPACERISKANALYZER_H
