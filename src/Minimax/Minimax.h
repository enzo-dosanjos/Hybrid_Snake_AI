/*************************************************************************
Minimax - Scores legal moves with alpha-beta search
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//---------- Interface of the module <Minimax> (file Minimax.h) ----------
#ifndef MINIMAX_H
#define MINIMAX_H

//------------------------------------------------------------------------
// Role of the <Minimax> module
// Search independent engine successors from the current player's perspective.
// The root player maximizes and every opponent minimizes the same score.
//------------------------------------------------------------------------

//--------------------------------------------------------- System Include
#include <map>

//-------------------------------------------------------- Used interfaces
#include "../GameEngine/GameEngine.h"
#include "SpaceRiskAnalyzer.h"

//------------------------------------------------------------------ Types
struct Rewards {
    double selfSpace = 2.72685;
    double oppSpace = 0.36796;

    double selfBlockRisk = -11.63637;
    double oppBlockRisk = -11.55511;

    double selfHeatRisk = -0.39336;
    double oppHeatRisk = -2.62494;

    double snakeLength = 1.66086;
};


struct MinimaxParams {
    Rewards rewards;
    // Count scheduled slots, including the root action, dead slots and forced deaths.
    int depth = 8;
};


//----------------------------------------------------------------- PUBLIC
class Minimax {
//--------------------------------------------------------- Public Methods
    public:
        std::map<Action, float> calculateMoveScores(const GameEngine &engine) const;
        // Contract : Return one exact search score per legal root action, or an empty map when no decision is possible.

//---------------------------------------------- Constructors - destructor
        explicit Minimax(const MinimaxParams &p_params, int lookahead = 5, float decay = 0.7f);
        // Contract : Require positive depth and finite reward weights.

//---------------------------------------------------------------- PRIVATE
    private:
        float evaluateState(const GameState &state, int rootPlayer) const;
        // Contract : Score the root's spatial metrics and length against the spatial metrics of living opponents.

        float minimax(const GameEngine &engine, int depth, float alpha, float beta, int rootPlayer) const;
        // Contract : Search the remaining scheduled slots using the supplied alpha-beta window.

        MinimaxParams params;
        SpaceRiskAnalyzer analyzer;
};

#endif //MINIMAX_H
