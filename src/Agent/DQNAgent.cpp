/*************************************************************************
DQNAgent - Selects a legal action using learned action values
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//--- Implementation of the module <DQNAgent> (file DQNAgent.cpp) ---

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <optional>

using namespace std;

//------------------------------------------------------- Personal Include
#include "DQNAgent.h"
#include "../GameEngine/DecisionEncoding.h"

//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
optional<Action> DQNAgent::selectAction(const GameEngine &engine)
// Algorithm : Encode the scheduled player's decision and delegate legal selection to the learner.
{
    int player = engine.getState().currentPlayer;
    if (engine.isTerminalFor(player) || engine.getActionMask(player).none())
    {
        return nullopt;
    }

    return dqn.selectAction(encodeDecision(engine, player, dqn.getFeatureSelection()), explore);
}  //----- end of selectAction


DQNAgent::DQNAgent(DQN &p_dqn, bool p_explore) : dqn(p_dqn), explore(p_explore)
// Algorithm : Retain the training learner and the requested exploration mode.
{
}  //----- end of DQNAgent
