/*************************************************************************
HybridAgent - todo
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//------ Interface of the module <HybridAgent> (file HybridAgent.h) ------
#ifndef HYBRIDAGENT_H
#define HYBRIDAGENT_H


//------------------------------------------------------------------------
// Role of the <HybridAgent> module
// todo
//------------------------------------------------------------------------

//-------------------------------------------------------- Used interfaces
#include "DQN.h"
#include "Minimax.h"
#include "Structs.h"

//-------------------------------------------------------------- Constants

//------------------------------------------------------------------ Types

//----------------------------------------------------------------- PUBLIC
class HybridAgent {
//--------------------------------------------------------- Public Methods
    public:
        std::string selectAction(
            const std::vector<float> &boardState,
            const std::vector<float> &extraFeatures,
            const Players &playerState,
            const std::deque<std::string> &last5Moves,
            int turn
        );

//---------------------------------------------- Constructors - destructor
        HybridAgent(
            int boardChannels,
            float lambda = 0.7,
            float eps = 1.0,
            float minEps = 0.01,
            float epsDecay = 0.995,
            int replayBufferSize = 100000
        ) : dqn(boardChannels, eps, minEps, epsDecay, replayBufferSize),
            minimax(boardChannels),
            lambda(lambda)
        {
            #ifdef MAP
                cout << "Calling the constructor of HybridAgent" << endl;
            #endif
        }

//---------------------------------------------------------------- PRIVATE
    private:
        DQN dqn;
        Minimax minimax;

        float lambda;
};


#endif //HYBRIDAGENT_H
