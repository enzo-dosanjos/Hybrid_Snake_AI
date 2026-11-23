/*************************************************************************
Agent - Common action-selection interface
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//---------- Interface of the module <Agent> (file Agent.h) ----------
#ifndef AGENT_H
#define AGENT_H

//------------------------------------------------------------------------
// Role of the <Agent> module
// Lets callers request a decision for the engine's scheduled player.
//------------------------------------------------------------------------

//--------------------------------------------------------- System Include
#include <optional>

//-------------------------------------------------------- Used interfaces
#include "../GameEngine/GameEngine.h"

//----------------------------------------------------------------- PUBLIC
class Agent {
//--------------------------------------------------------- Public Methods
    public:
        virtual std::optional<Action> selectAction(const GameEngine &engine) = 0;
        // Contract : Return a legal action for the scheduled player without changing the engine.
        // Return no action for a dead or trapped player or a finished game.

//---------------------------------------------- Constructors - destructor
        virtual ~Agent() = default;
};

#endif //AGENT_H
