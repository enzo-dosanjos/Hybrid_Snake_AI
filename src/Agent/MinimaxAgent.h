/*************************************************************************
MinimaxAgent - Selects a legal action using Minimax scores
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//------ Interface of the module <MinimaxAgent> (file MinimaxAgent.h) ------
#ifndef MINIMAXAGENT_H
#define MINIMAXAGENT_H

//------------------------------------------------------------------------
// Role of the <MinimaxAgent> module
// Choose the highest-scoring legal action for the scheduled player.
//------------------------------------------------------------------------

//-------------------------------------------------------- Used interfaces
#include "Agent.h"
#include "../Minimax/Minimax.h"

//----------------------------------------------------------------- PUBLIC
class MinimaxAgent : public Agent {
//--------------------------------------------------------- Public Methods
    public:
        std::optional<Action> selectAction(const GameEngine &engine) override;
        // Contract : Break score ties by Action enum order and return no action when there are no scores.

//---------------------------------------------- Constructors - destructor
        explicit MinimaxAgent(const MinimaxParams &params);

//---------------------------------------------------------------- PRIVATE
    private:
        Minimax minimax;
};

#endif //MINIMAXAGENT_H
