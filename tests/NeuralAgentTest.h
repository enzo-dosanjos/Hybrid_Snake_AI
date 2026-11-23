/*************************************************************************
NeuralAgentTest - Shared learner and game fixtures for agent tests
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//---- Interface of the module <NeuralAgentTest> (file NeuralAgentTest.h) ----
#ifndef NEURALAGENTTEST_H
#define NEURALAGENTTEST_H

//------------------------------------------------------------------------
// Role of the <NeuralAgentTest> module
// Supply comparable scenarios for the neural and hybrid policies.
//------------------------------------------------------------------------

//-------------------------------------------------------- Used interfaces
#include "DQNTest.h"
#include "src/Agent/Agent.h"

//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
inline ModelConfig makeAgentModelConfig()
// Algorithm : Configure a small model with a non-default selection of optional features.
{
    FeatureSelection features{false, true, false, false, true};
    auto config = makeModelConfig(StateAnalyzer::getFeatureCount(2, features));
    config.features = features;
    config.convolutionLayers = {{"ReLU", 3, 1, 1, 3}};
    config.denseHiddenSizes.clear();
    config.minEpsilon = 0.f;
    config.epsilonDecay = 0.f;
    return config;
}  //----- end of makeAgentModelConfig


inline GameEngine makeAgentDecision()
// Algorithm : Schedule player two on a wall with three legal moves and player one configured for local control.
{
    GameEngine engine({4, 4, 2, 2, 1}, {{0, 0}, {0, 2}, {3, 1}});
    engine.finishTurn();
    return engine;
}  //----- end of makeAgentDecision


inline EncodedState referenceDecision(const GameEngine &engine, int player,
                                      const FeatureSelection &features)
// Algorithm : Build expected network inputs directly from the three encoder contracts.
{
    const auto &state = engine.getState();
    return {
        Observation{}.encode(state, player),
        StateAnalyzer{}.computeExtraFeatures(state, player, features),
        engine.getActionMask(player)
    };
}  //----- end of referenceDecision


inline void checkNoAgentDecision(Agent &agent)
// Algorithm : Check that trapped, finished and dead-slot snapshots produce no action and are preserved.
{
    GameEngine trapped({2, 2, 1, 3, 1}, {{0, 0}, {0, 0}, {1, 0}, {0, 1}});
    GameEngine finished({3, 3, 1, 2, 1}, {{0, 0}, {0, 0}, {2, 2}});
    finished.playerDied(2);
    GameEngine deadSlot({3, 3, 1, 3, 1}, {{0, 0}, {0, 0}, {2, 2}, {0, 2}});
    deadSlot.playerDied(1);

    check(!trapped.isTerminalState() && trapped.getActionMask(1).none(), "Invalid trapped fixture");
    check(finished.isTerminalState() && finished.getActionMask(1).any(), "Invalid finished fixture");
    check(!deadSlot.isTerminalState() && !deadSlot.isAlive(1), "Invalid dead-slot fixture");

    for (const auto *engine : {&trapped, &finished, &deadSlot})
    {
        auto before = engine->getState();
        check(!agent.selectAction(*engine), "An unavailable decision produced an action");
        checkState(engine->getState(), before);
    }
}  //----- end of checkNoAgentDecision

#endif //NEURALAGENTTEST_H
