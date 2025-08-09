/*************************************************************************
Reward - utility functions for the game processing
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//------------ Interface of the module <Reward> (file Reward.h) ------------
#ifndef UTILS_H
#define UTILS_H

//------------------------------------------------------------------------
// Role of the <Reward> module
// todo
//------------------------------------------------------------------------

//-------------------------------------------------------- Used interfaces
#include "GameEngine.h"
#include "Structs.h"

//------------------------------------------------------------------ Types
// Reward configuration
struct NNRewards {
    const float winningMove;
    const float losingMove;

    // Aggression rewards
    const float killingSetupMove;       // When the opps have less than 2 escape moves
    const float killZoneEnterMove;      // When moving an opp into a kill zone (close to corner/walls)
    const float escapeReductionMove;    // Per opponent escape route closed

    // Anti-passivity
    const float entropyPenalty;         // For low action diversity

    const float survivingMove;

    NNRewards(
        int W,
        int H
    ) :
        winningMove(2.0 * W * H),
        losingMove(-2.0 * W * H),
        killingSetupMove(15.0 * sqrt(W*H)),
        killZoneEnterMove(6.0 * sqrt(W*H)),
        escapeReductionMove(2.4 * sqrt(W*H)),
        entropyPenalty(-3.0 * sqrt(W*H)),
        survivingMove(0.1 * sqrt(W*H))
    {}  // todo: normalise rewards
};

//----------------------------------------------------------------- PUBLIC
class Reward
{
//--------------------------------------------------------- Public Methods
    public:
        float computeReward(
                const std::vector<float> &boardState,
                const std::vector<float> &extraState,
                const std::vector<float> &prevExtraState,
                const PlayersData &playerState,
                int playerNum,
                int turn,
                bool isTerminalState,
                bool isAlive
            );
        // Usage :
        //
        // Contract :
        //

//---------------------------------------------- Constructors - destructor
        Reward(
            GameConfig config
        ) : rewards(config.W, config.H), numOpponents(config.N-1)
        // Usage :
        //
        // Contract :
        //
        {
            #ifdef MAP
                cout << "Calling the constructor of Reward" << endl;
            #endif
            maxTurns = (config.M() > 1) ? 0.8*(config.W*config.H()*config.M) : (config.W*config.H*config.M);
        };


//---------------------------------------------------------------- PRIVATE
    private:
        NNRewards rewards;

        int maxTurns;
        int numOpponents;
};



#endif //UTILS_H
