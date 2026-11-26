/*************************************************************************
RandomLegalAgent - Samples legal Snake moves uniformly
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//--------- Interface of the module <RandomLegalAgent> (file RandomLegalAgent.h) ---------

#ifndef RANDOMLEGALAGENT_H
#define RANDOMLEGALAGENT_H

//------------------------------------------------------------------------
// Role of the <RandomLegalAgent> module
// Sample actions using the generator owned by the experiment.
//------------------------------------------------------------------------

//--------------------------------------------------------- System Include
#include <random>

//-------------------------------------------------------- Used interfaces
#include "../../src/Agent/Agent.h"

//----------------------------------------------------------------- PUBLIC
class RandomLegalAgent : public Agent {
//--------------------------------------------------------- Public Methods
    public:
        std::optional<Action> selectAction(const GameEngine &engine) override;
        // Contract : Sample uniformly from the legal mask. Return no action when the scheduled player has no legal moves.

//---------------------------------------------- Constructors - destructor
        explicit RandomLegalAgent(std::mt19937 &generator);
        // Contract : The caller owns the generator and may reseed it between episodes.

//---------------------------------------------------------------- PRIVATE
    private:
        std::mt19937 &generator;
};

#endif //RANDOMLEGALAGENT_H
