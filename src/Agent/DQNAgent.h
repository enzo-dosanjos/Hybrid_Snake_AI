/*************************************************************************
DQNAgent - Selects a legal action using learned action values
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//-------- Interface of the module <DQNAgent> (file DQNAgent.h) --------
#ifndef DQNAGENT_H
#define DQNAGENT_H

//------------------------------------------------------------------------
// Role of the <DQNAgent> module
// Apply the learner's policy to the scheduled player's decision.
//------------------------------------------------------------------------

//-------------------------------------------------------- Used interfaces
#include "Agent.h"
#include "../DQN/DQN.h"

//----------------------------------------------------------------- PUBLIC
class DQNAgent : public Agent {
//--------------------------------------------------------- Public Methods
    public:
        std::optional<Action> selectAction(const GameEngine &engine) override;
        // Contract : Choose an epsilon-greedy action in training mode and a greedy action in evaluation mode.

//---------------------------------------------- Constructors - destructor
        DQNAgent(DQN &p_dqn, bool p_explore);
        // Contract : The borrowed learner must outlive this agent.

//---------------------------------------------------------------- PRIVATE
    private:
        DQN &dqn;
        bool explore;
};

#endif //DQNAGENT_H
