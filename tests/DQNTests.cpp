/*************************************************************************
DQNTests - Legal actions, Bellman targets and learning
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//----- Implementation of the module <DQNTests> (file DQNTests.cpp) ------

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <algorithm>
#include <set>

using namespace std;

//------------------------------------------------------- Personal Include
#include "DQNTest.h"


//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
void testActionMasks()
// Algorithm : Tests exploratory and greedy legality by exercising all sixteen possible action masks.
{
    GameEngine engine({5, 4, 2, 2, 1}, {{0, 0}, {1, 1}, {4, 3}});
    auto state = encodeState(engine, 1);
    DQN learner(makeModelConfig(state.extraFeatures.size()), mt19937(42));

    for (int bits = 0; bits < 16; bits++)
    {
        state.legalActions = bitset<4>(bits);

        for (bool explore : {false, true})
        {
            for (int attempt = 0; attempt < 16; attempt++)
            {
                auto action = learner.selectAction(state, explore);

                if (bits == 0)
                {
                    check(!action, "Empty mask produced an action");
                }
                else
                {
                    check(action.has_value(), "Nonempty mask produced no action");
                    check(state.legalActions[static_cast<int>(*action)], "Policy selected an illegal action");
                }
            }
        }
    }
}  //----- end of testActionMasks


void testGreedySelection()
// Algorithm : Tests greedy choice by comparing the selected action with every legal Q-value.
{
    GameEngine engine({5, 4, 2, 2, 1}, {{0, 0}, {1, 1}, {4, 3}});
    auto state = encodeState(engine, 1);
    DQN learner(makeModelConfig(state.extraFeatures.size()), mt19937(42));

    for (int bits = 1; bits < 16; bits++)
    {
        state.legalActions = bitset<4>(bits);
        auto values = learner.forwardPropagation(state);
        auto action = learner.selectAction(state, false);
        check(action.has_value(), "Greedy selection returned no action");

        for (int candidate = 0; candidate < 4; candidate++)
        {
            if (state.legalActions[candidate])
            {
                check(values[static_cast<int>(*action)] >= values[candidate], "Greedy action is not maximal");
            }
        }
    }
}  //----- end of testGreedySelection


void testTerminalTargets()
// Algorithm : Tests terminal bootstrap exclusion by training identical networks with different terminal next states.
{
    GameEngine engine({5, 4, 2, 2, 1}, {{0, 0}, {1, 1}, {4, 3}});
    auto state = encodeState(engine, 1);
    auto alternate = state;
    alternate.extraFeatures.assign(state.extraFeatures.size(), 0.25f);
    alternate.legalActions.reset();
    auto config = makeModelConfig(state.extraFeatures.size());
    DQN first(config, mt19937(42));
    DQN second(config, mt19937(42));

    for (int index = 0; index < 8; index++)
    {
        first.remember(Transition{state, Action::Down, 1.f, state, true});
        second.remember(Transition{state, Action::Down, 1.f, alternate, true});
    }

    // Identical terminal rewards must produce identical updates regardless of next-state values or masks.
    first.train(4, 0.001f);
    second.train(4, 0.001f);
    checkPredictions(first, second, state);
}  //----- end of testTerminalTargets


void testLearningFromEpisode()
// Algorithm : Tests learning integration by training on an engine-generated terminal transition and checking error reduction.
{
    GameEngine engine({4, 4, 1, 2, 1}, {{0, 0}, {0, 0}, {1, 0}});
    auto state = encodeState(engine, 1);
    DQN learner(makeModelConfig(state.extraFeatures.size()), mt19937(42));
    engine.updateStep(Action::Down);
    engine.updateStep(Action::Up);
    check(engine.isTerminalFor(1) && engine.isAlive(1), "Episode fixture must end with a learner win");
    auto next = encodeState(engine, 1);
    int action = static_cast<int>(Action::Down);
    auto before = learner.forwardPropagation(state);
    float reward = before[action] + 1.f;

    for (int index = 0; index < 8; index++)
    {
        learner.remember(Transition{state, Action::Down, reward, next, true});
    }

    for (int update = 0; update < 5; update++)
    {
        learner.train(4, 0.001f);
    }

    auto after = learner.forwardPropagation(state);

    for (float value : after)
    {
        check(isfinite(value), "Training produced non-finite predictions");
    }

    check(abs(after[action] - reward) < abs(before[action] - reward), "Training did not reduce terminal error");
}  //----- end of testLearningFromEpisode


void testSeededTraining()
// Algorithm : Tests reproducible learning by comparing equally seeded models through replay updates and target synchronization.
{
    GameEngine engine({5, 4, 2, 2, 1}, {{0, 0}, {1, 1}, {4, 3}});
    auto state = encodeState(engine, 1);
    auto config = makeModelConfig(state.extraFeatures.size());
    DQN first(config, mt19937(42));
    DQN second(config, mt19937(42));
    populateReplay(first, state);
    populateReplay(second, state);

    for (int update = 0; update < 8; update++)
    {
        first.train(5, 0.001f);
        second.train(5, 0.001f);
        checkPredictions(first, second, state);

        if (update % 3 == 0)
        {
            first.updateTargetNetworks();
            second.updateTargetNetworks();
        }
    }
}  //----- end of testSeededTraining


vector<TestCase> getTests()
// Algorithm : Registers this component’s named test functions for independent selection and execution.
{
    return {
        {"action_masks", testActionMasks},
        {"greedy_selection", testGreedySelection},
        {"terminal_targets", testTerminalTargets},
        {"learning_from_episode", testLearningFromEpisode},
        {"seeded_training", testSeededTraining}
    };
}  //----- end of getTests
