/*************************************************************************
Reward - utility functions for
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//-------- Implementation of the module <Reward> (file Reward.cpp) ---------

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace std;

//------------------------------------------------------- Personal Include
#include "../include/Reward.h"


//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
float Reward::computeReward(const vector<float> &boardState, const vector<float> &extraState,  // todo: split into methods for each rewards
                          const vector<float> &prevExtraState, const PlayersData &playerState,
                          int playerNum, int turn, bool isTerminalState, bool isAlive)
// Algorithm : Compute the mono-objective reward for a given state
{
    // - If my snake's head collides with a border or its own body, return negative lot
    // - If the opponent's snake head collides, return a positive lot
    // - Otherwise, return a small positive reward for survival
    float score = 0.0;

    float winningMove =  rewards.winningMove * (1.0 + turn/static_cast<float>(maxTurns));  // Scale the winning reward with the turn number, so it is more rewarding in the late game
    float losingMove =  rewards.losingMove * (0.2 + 0.8*pow(turn/static_cast<float>(maxTurns), 2));  // Curve the penalty for losing, so it is less severe in the early game

    if (isAlive && isTerminalState)
    {
            score += winningMove;
    }
    else if (isTerminalState)
    {
        score += losingMove;
    }
    else
    {
        for (const auto &player : playerState)
        {
            if (player.first != playerNum)
            {
                if (!gameEngine.isAlive(playerState, boardState, player.first))
                {
                    score += winningMove;
                }
            }
        }
    }

    if (oppWillDie || oppDied)  // todo: check if it's my ai that killed the opp (works in 1v1)
    {
        score += winningMove;
    }

    score += rewards.killingSetupMove * nbOppKillSetup;

    const int nbOpp = (gameEngine.getPlayers().size()-1);
    const int ESCAPE_ROUTES_INDEX = 6*nbOpp + 3;
    const int KILL_ZONE_INDEX = ESCAPE_ROUTES_INDEX + nbOpp;
    const int PHASE_INDEX = KILL_ZONE_INDEX + nbOpp;
    const int ENTROPY_INDEX = PHASE_INDEX + 1;

    // Aggression rewards
    float aggressionScore = 0.0;

    for (int i = 0; i < gameEngine.getPlayers().size()-1; i++)
    {
        float killZoneValue = 1.0 / (1.0 + exp(-4.0*(extraState[KILL_ZONE_INDEX + i]-0.3)));  // sigmoid scaling for the killzone value
        aggressionScore += rewards.killZoneEnterMove * killZoneValue;
    }

    for (int i = 0; i < gameEngine.getPlayers().size()-1; i++)
    {
        if(prevExtraState[ESCAPE_ROUTES_INDEX + i] - extraState[ESCAPE_ROUTES_INDEX + i] > 0)
        {
            aggressionScore += rewards.escapeReductionMove * (prevExtraState[ESCAPE_ROUTES_INDEX + i] - extraState[ESCAPE_ROUTES_INDEX + i]);
        }
    }

    // Entropy penalty
    const float minAcceptableEntropy = 0.7; // ~3 moves used
    if(extraState[ENTROPY_INDEX] < minAcceptableEntropy) {
        aggressionScore += ((minAcceptableEntropy - extraState[ENTROPY_INDEX])/minAcceptableEntropy) * rewards.entropyPenalty;
    }

    // Depends on the size for exponential reward
    score += rewards.survivingMove * gameEngine.getPlayers().at(playerNum).first.size();

    // Z-score normalization
    static float reward_mean = 0.0f;
    static float reward_std = 1.0;
    float z_score = ((score + aggressionScore) - reward_mean) / reward_std;
    reward_mean = 0.99f * reward_mean + 0.01f * (score + aggressionScore);
    reward_std = 0.99f * reward_std + 0.01f * fabs((score + aggressionScore) - reward_mean);
    return z_score;
}  //----- End of computeReward
