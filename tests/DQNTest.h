/*************************************************************************
DQNTest - Provides model configuration, encoding and replay fixtures
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//--------- Interface of the module <DQNTest> (file DQNTest.h) ----------

#ifndef DQNTEST_H
#define DQNTEST_H

//------------------------------------------------------------------------
// Role of the <DQNTest> module
// Provides model configuration, encoding and replay fixtures
//------------------------------------------------------------------------

//--------------------------------------------------------- System Include
#include <random>

//-------------------------------------------------------- Used interfaces
#include "GameTest.h"
#include "src/DQN/DQN.h"
#include "src/GameEngine/Observation.h"
#include "src/GameEngine/StateAnalyzer.h"


//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
inline EncodedState encodeState(GameEngine &engine, int player)
// Algorithm : Builds the model input from the engine’s board, analytical features and legal-action mask.
{
    return {
        Observation{}.encode(engine.getState(), player),
        StateAnalyzer{}.computeExtraFeatures(engine.getState(), player),
        engine.getActionMask(player)
    };
}  //----- end of encodeState


inline ModelConfig makeModelConfig(int featureCount)
// Algorithm : Builds a small deterministic-test configuration with fixed replay, discount and exploration settings.
{
    ModelConfig config;
    config.numPlayers = 2;
    config.extraFeatureCount = featureCount;
    config.replayCapacity = 32;
    config.epsilon = 1.f;
    config.gamma = 0.9f;

    return config;
}  //----- end of makeModelConfig


inline void checkPredictions(DQN &first, DQN &second, const EncodedState &state)
// Algorithm : Checks model equivalence by comparing all four Q-values on the same encoded state.
{
    auto left = first.forwardPropagation(state);
    auto right = second.forwardPropagation(state);

    for (int action = 0; action < 4; action++)
    {
        checkNear(left[action], right[action], "DQN prediction differs", 1e-6f, 1e-5f);
    }
}  //----- end of checkPredictions


inline void populateReplay(DQN &learner, const EncodedState &state)
// Algorithm : Fills replay with varied rewards and terminal flags so continuation checks exercise both target paths.
{
    for (int index = 0; index < 24; index++)
    {
        learner.remember(Transition{
            state, Action::Down, 0.1f * index - 1.f, state, index % 3 == 0
        });
    }
}  //----- end of populateReplay

#endif //DQNTEST_H
