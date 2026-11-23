/*************************************************************************
DecisionEncoding - Assembles a model input from the engine
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//---- Interface of the module <DecisionEncoding> (file DecisionEncoding.h) ----
#ifndef DECISIONENCODING_H
#define DECISIONENCODING_H

//------------------------------------------------------------------------
// Role of the <DecisionEncoding> module
// Keep board, feature and action-mask perspectives aligned for inference and replay.
//------------------------------------------------------------------------

//-------------------------------------------------------- Used interfaces
#include "GameEngine.h"
#include "Observation.h"
#include "StateAnalyzer.h"

//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
inline EncodedState encodeDecision(const GameEngine &engine, int player,
                                   const FeatureSelection &features)
// Algorithm : Assemble the player's board, selected extra features and legal-action mask from the same engine.
{
    const GameState &state = engine.getState();
    return {
        Observation{}.encode(state, player),
        StateAnalyzer{}.computeExtraFeatures(state, player, features),
        engine.getActionMask(player)
    };
}  //----- end of encodeDecision

#endif //DECISIONENCODING_H
