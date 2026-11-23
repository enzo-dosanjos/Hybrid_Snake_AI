/*************************************************************************
MinimaxAgent - Selects a legal action using Minimax scores
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//--- Implementation of the module <MinimaxAgent> (file MinimaxAgent.cpp) ---

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <iterator>

using namespace std;

//------------------------------------------------------- Personal Include
#include "MinimaxAgent.h"

//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
optional<Action> MinimaxAgent::selectAction(const GameEngine &engine)
// Algorithm : Choose the highest score, keeping the first action in map order on ties.
{
    auto scores = minimax.calculateMoveScores(engine);

    if (scores.empty())
    {
        return nullopt;
    }

    auto best = scores.begin();

    for (auto candidate = next(best); candidate != scores.end(); ++candidate)
    {
        if (candidate->second > best->second)
        {
            best = candidate;
        }
    }

    return best->first;
}  //----- end of selectAction


MinimaxAgent::MinimaxAgent(const MinimaxParams &params) : minimax(params)
// Algorithm : Configure the search with the supplied Minimax parameters.
{
}  //----- end of MinimaxAgent
