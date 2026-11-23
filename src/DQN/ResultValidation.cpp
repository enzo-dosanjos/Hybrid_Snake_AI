/*************************************************************************
ResultValidation - Evaluates agents on configured Snake matches
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//-- Implementation of the module <ResultValidation> (file ResultValidation.cpp) --

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <stdexcept>

using namespace std;

//------------------------------------------------------- Personal Include
#include "ResultValidation.h"

//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
ValidationResult ResultValidation::validate(Agent &agent, const vector<ValidationMatch> &validationSet) const
// Algorithm : Play each caller-supplied matchup through the engine and aggregate wins and survival turns.
{
    ValidationResult result;
    double survivalTurns = 0.0;

    for (const auto &match : validationSet)
    {
        GameEngine engine(match.config, match.origins);
        int evaluatedPlayer = engine.getP();
        if (match.opponents.size() != static_cast<size_t>(engine.getN() - 1))
        {
            throw invalid_argument("Validation requires an opponent for each other seat");
        }

        while (!engine.isTerminalFor(evaluatedPlayer))
        {
            int player = engine.getState().currentPlayer;
            if (!engine.isAlive(player))
            {
                engine.finishTurn();
                continue;
            }

            survivalTurns++;
            ActionMask legalActions = engine.getActionMask(player);
            if (legalActions.none())
            {
                engine.playerDied(player);
                engine.finishTurn();
                continue;
            }

            // Opponents occupy every seat except the evaluated player's seat.
            int opponentIndex = player - 1 - (player > evaluatedPlayer);
            Agent &policy = player == evaluatedPlayer ? agent : match.opponents[opponentIndex].get();
            auto action = policy.selectAction(engine);
            if (!action || static_cast<size_t>(*action) >= legalActions.size() ||
                !legalActions[static_cast<size_t>(*action)])
            {
                throw logic_error("Validation agent did not select a legal action");
            }

            engine.updateStep(*action);
        }

        result.matches++;
        if (engine.isAlive(evaluatedPlayer))
        {
            result.wins++;
        }
    }

    if (result.matches > 0)
    {
        result.meanSurvivalTurns = survivalTurns / result.matches;
    }

    return result;
}  //----- end of validate


float ResultValidation::calculateScore(const ValidationResult &result) const
// Algorithm : Divide the number of wins by the number of matches when the set is nonempty.
{
    return result.matches == 0 ? 0.f : static_cast<float>(result.wins) / result.matches;
}  //----- end of calculateScore
