/*************************************************************************
TrainingLoop - Plays learner decisions and their intervening turns
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//------ Interface of the module <TrainingLoop> (file TrainingLoop.h) ------
#ifndef TRAININGLOOP_H
#define TRAININGLOOP_H

//------------------------------------------------------------------------
// Role of the <TrainingLoop> module
// Produce a replay transition from real turns played in the shared engine.
//------------------------------------------------------------------------

//--------------------------------------------------------- System Include
#include <functional>
#include <optional>
#include <vector>

//-------------------------------------------------------- Used interfaces
#include "Agent/Agent.h"
#include "DQN/Reward.h"

//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
std::optional<Transition> playLearnerDecision(
    GameEngine &engine, Agent &behavior,
    const std::vector<std::reference_wrapper<Agent>> &opponents,
    const Reward &reward, const FeatureSelection &features
);
// Contract : Use the controlled seat as the learner. Opponents are in ascending seat order.
// Advance to a learner decision or termination. Return no transition when no action was executed.

#endif //TRAININGLOOP_H
