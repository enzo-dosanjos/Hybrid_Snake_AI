/*************************************************************************
Reward - Scores the interval between learner decisions
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//------------ Interface of the module <Reward> (file Reward.h) ------------
#ifndef REWARD_H
#define REWARD_H

//------------------------------------------------------------------------
// Role of the <Reward> module
// Scores outcomes and opponent pressure using authoritative game snapshots.
//------------------------------------------------------------------------

//-------------------------------------------------------- Used interfaces
#include "../Structs.h"

//------------------------------------------------------------------ Types
struct NNRewards {
    const float winningMove;
    const float losingMove;
    const float killingSetupMove;
    const float killZoneEnterMove;
    const float escapeReductionMove;
    const float survivingMove;

    NNRewards(int W, int H);
    // Contract : Scale the fixed reward weights using the board dimensions.
};

//----------------------------------------------------------------- PUBLIC
class Reward {
//--------------------------------------------------------- Public Methods
    public:
        float computeReward(const GameState &previous, const GameState &next, int player) const;
        // Contract : Pass the living learner's decision snapshot and the next decision
        // or termination snapshot from the same game. Both must contain current masks.

//---------------------------------------------- Constructors - destructor
        explicit Reward(const GameConfig &p_config);
        // Contract : Use the same configuration as the supplied snapshots.

//---------------------------------------------------------------- PRIVATE
    private:
        GameConfig config;
        NNRewards rewards;
};

#endif //REWARD_H
