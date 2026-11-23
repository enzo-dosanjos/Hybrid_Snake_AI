/*************************************************************************
HybridAgent - Combines Minimax and learned action scores
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//------ Interface of the module <HybridAgent> (file HybridAgent.h) ------
#ifndef HYBRIDAGENT_H
#define HYBRIDAGENT_H

//------------------------------------------------------------------------
// Role of the <HybridAgent> module
// Weight normalized search and neural scores for the scheduled player.
//------------------------------------------------------------------------

//--------------------------------------------------------- System Include
#include <array>

//-------------------------------------------------------- Used interfaces
#include "Agent.h"
#include "../DQN/DQN.h"
#include "../Minimax/Minimax.h"

//----------------------------------------------------------------- PUBLIC
class HybridAgent : public Agent {
//--------------------------------------------------------- Public Methods
    public:
        std::optional<Action> selectAction(const GameEngine &engine) override;
        // Contract : Maximize the weighted legal scores, breaking ties in Action order.
        // Training mode may take a uniform legal action at the learner's current epsilon.

//---------------------------------------------- Constructors - destructor
        HybridAgent(DQN &p_dqn, const MinimaxParams &params, float p_lambda, bool p_explore);
        // Contract : Lambda is the neural score weight in [0, 1]. The learner must outlive this agent.

//---------------------------------------------------------------- PRIVATE
    private:
        static std::array<double, 4> normalizeLegalScores(const std::array<float, 4> &scores,
                                                         const ActionMask &mask);
        // Contract : Scale only legal scores to [0, 1]; a constant scorer contributes zero.

        DQN &dqn;
        Minimax minimax;
        float lambda;
        bool explore;
};

#endif //HYBRIDAGENT_H
