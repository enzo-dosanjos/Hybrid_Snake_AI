/*************************************************************************
ResultValidation - Evaluates agents on configured Snake matches
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//---- Interface of the module <ResultValidation> (file ResultValidation.h) ----
#ifndef RESULT_VALIDATION_H
#define RESULT_VALIDATION_H

//------------------------------------------------------------------------
// Role of the <ResultValidation> module
// Play evaluation matches using the same engine and policies as other callers.
//------------------------------------------------------------------------

//--------------------------------------------------------- System Include
#include <cstddef>
#include <functional>
#include <vector>

//-------------------------------------------------------- Used interfaces
#include "../Agent/Agent.h"

//------------------------------------------------------------------ Types
struct ValidationMatch {
    // P is the evaluated seat. Origins use the engine's one-based player IDs.
    GameConfig config;
    std::vector<Coord> origins;
    // Opponents are in ascending player ID order, omitting P.
    std::vector<std::reference_wrapper<Agent>> opponents;
};


struct ValidationResult {
    std::size_t matches = 0;
    std::size_t wins = 0;
    // Count living players' turns through the evaluated player's death or the end of the game.
    double meanSurvivalTurns = 0.0;
};


//----------------------------------------------------------------- PUBLIC
class ResultValidation {
//--------------------------------------------------------- Public Methods
    public:
        ValidationResult validate(Agent &agent, const std::vector<ValidationMatch> &validationSet) const;
        // Contract : Pass evaluation-mode agents for all seats, with exploration disabled.

        float calculateScore(const ValidationResult &result) const;
        // Contract : Return the fraction of matches won, or zero for an empty set.
};

#endif //RESULT_VALIDATION_H
