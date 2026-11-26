/*************************************************************************
RandomLegalAgent - Samples legal Snake moves uniformly
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//--------- Implementation of the module <RandomLegalAgent> (file RandomLegalAgent.cpp) ---------

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <vector>

using namespace std;

//------------------------------------------------------- Personal Include
#include "RandomLegalAgent.h"

//----------------------------------------------------------------- PUBLIC
//---------------------------------------------- Constructors - destructor
RandomLegalAgent::RandomLegalAgent(mt19937 &generator) : generator(generator)
// Algorithm : Borrow the generator used for random moves during this run.
{
}  //----- end of RandomLegalAgent


//--------------------------------------------------------- Public Methods
optional<Action> RandomLegalAgent::selectAction(const GameEngine &engine)
// Algorithm : Sample uniformly among the scheduled player's legal actions.
{
    auto mask = engine.getActionMask(engine.getState().currentPlayer);
    vector<Action> actions;

    for (size_t i = 0; i < mask.size(); i++)
    {
        if (mask[i])
        {
            actions.push_back(static_cast<Action>(i));
        }
    }

    if (actions.empty())
    {
        return nullopt;
    }

    return actions[uniform_int_distribution<size_t>(0, actions.size() - 1)(generator)];
}  //----- end of selectAction
