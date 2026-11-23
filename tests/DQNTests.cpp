/*************************************************************************
DQNTests - Legal actions, Bellman targets and learning
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//----- Implementation of the module <DQNTests> (file DQNTests.cpp) ------

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <algorithm>
#include <utility>

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


void testExplorationAndDecay()
// Algorithm : Check that full exploration covers the mask, then that a replay update can disable it.
{
    GameEngine engine({5, 4, 2, 2, 1}, {{0, 0}, {1, 1}, {4, 3}});
    auto state = encodeState(engine, 1);
    state.legalActions = ActionMask(13);
    auto config = makeModelConfig(state.extraFeatures.size());
    config.minEpsilon = 0.f;
    config.epsilonDecay = 0.f;
    DQN learner(config, mt19937(42));

    learner.train(4, 0.01f);
    ActionMask explored;
    for (int attempt = 0; attempt < 64; attempt++)
    {
        auto action = learner.selectAction(state, true);
        check(action.has_value(), "Exploratory policy returned no action");
        explored.set(static_cast<int>(*action));
    }

    check(explored == state.legalActions, "Empty replay changed exploration or exploration missed legal moves");

    learner.remember(Transition{state, Action::Down, 1.f, {}, true});
    learner.train(4, 0.01f);
    auto greedy = learner.selectAction(state, false);
    for (int attempt = 0; attempt < 64; attempt++)
    {
        check(learner.selectAction(state, true) == greedy, "Exploration continued after epsilon decayed to zero");
    }
}  //----- end of testExplorationAndDecay


void testTerminalTargets()
// Algorithm : Tests terminal bootstrap exclusion by training identical networks with different terminal next states.
{
    GameEngine engine({5, 4, 2, 2, 1}, {{0, 0}, {1, 1}, {4, 3}});
    auto state = encodeState(engine, 1);
    auto alternate = state;
    alternate.board.clear();
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


void testEmptyNextActionMask()
// Algorithm : Compare a nonterminal empty next mask with a terminal transition using the same reward.
{
    GameEngine engine({5, 4, 2, 2, 1}, {{0, 0}, {1, 1}, {4, 3}});
    auto state = encodeState(engine, 1);
    auto config = makeModelConfig(state.extraFeatures.size());
    DQN learner(config, mt19937(42));
    DQN reference(config, mt19937(42));

    // The empty mask must prevent evaluation even when next-state inputs are absent.
    learner.remember(Transition{state, Action::Down, 1.f, {}, false});
    reference.remember(Transition{state, Action::Down, 1.f, state, true});
    learner.train(1, 0.01f);
    reference.train(1, 0.01f);
    checkPredictions(learner, reference, state);
}  //----- end of testEmptyNextActionMask


void testLegalTargetsAndSynchronization()
// Algorithm : Compare bootstrapped updates with explicit terminal targets before and after synchronization.
{
    GameEngine engine({5, 4, 2, 2, 1}, {{0, 0}, {1, 1}, {4, 3}});
    auto state = encodeState(engine, 1);
    auto config = makeModelConfig(state.extraFeatures.size());
    config.replayCapacity = 1;
    config.convolutionLayers = {{"linear", 1, 1, 0, 1}};
    config.denseHiddenSizes.clear();
    DQN learner(config, mt19937(42));
    DQN reference(config, mt19937(42));
    auto originalValues = reference.forwardPropagation(state);

    // Keep the two lowest Q-values legal so an unmasked maximum gives a different target.
    array<pair<float, int>, 4> ranked;
    for (size_t action = 0; action < originalValues.size(); action++)
    {
        ranked[action] = {originalValues[action], action};
    }

    sort(ranked.begin(), ranked.end());
    int legalFirst = ranked[0].second;
    int legalBest = ranked[1].second;
    check(ranked.back().first - ranked[1].first > 1e-5f && ranked[1].first - ranked[0].first > 1e-5f,
          "Fixture must distinguish the legal and unrestricted maxima");
    auto next = state;
    next.legalActions.reset();
    next.legalActions.set(legalFirst);
    next.legalActions.set(legalBest);

    auto action = static_cast<Action>(legalBest);
    state.legalActions.set(legalBest);
    float reward = 1.f;
    learner.remember(Transition{state, action, reward, next, false});
    reference.remember(Transition{state, action, reward + config.gamma * originalValues[legalBest], {}, true});

    for (int update = 0; update < 2; update++)
    {
        learner.train(1, 0.1f);
        reference.train(1, 0.1f);
        checkPredictions(learner, reference, state);
    }

    auto updatedValues = reference.forwardPropagation(state);
    float updatedMaximum = max(updatedValues[legalFirst], updatedValues[legalBest]);
    check(abs(updatedMaximum - originalValues[legalBest]) > 1e-5f,
          "Fixture must distinguish the original and synchronized targets");
    learner.updateTargetNetworks();
    reference.remember(Transition{state, action, reward + config.gamma * updatedMaximum, {}, true});
    learner.train(1, 0.1f);
    reference.train(1, 0.1f);
    checkPredictions(learner, reference, state);
}  //----- end of testLegalTargetsAndSynchronization


void testPooledBatchUpdate()
// Algorithm : Check a linear model's predictions against a hand-derived batch update on a rectangular board.
{
    FeatureSelection selection{false, false, false, false, false};
    auto config = makeModelConfig(StateAnalyzer::getFeatureCount(2, selection));
    config.features = selection;
    config.convolutionLayers = {{"linear", 1, 1, 0, 1}};
    config.denseHiddenSizes.clear();
    int channels = CHANNELS_PER_PLAYER * config.numPlayers + 1;
    mt19937 seed(29);
    DQN learner(config, seed);

    // Use the primitives only to recover the seeded parameters. The forward and update equations below are independent.
    CNN convolution(channels, seed);
    convolution.addLayer(config.convolutionLayers.front());
    auto convLayer = convolution.getNN().front();
    FCNN dense(1 + config.extraFeatureCount, seed);
    dense.addLayer("linear", 4);
    auto denseLayer = dense.getNN().front();

    EncodedState state{BoardTensor(channels, Matrix<float>(3, 2)),
                       vector<float>(config.extraFeatureCount), ActionMask(8)};
    state.board.front()[1][2] = 12.f;
    state.extraFeatures[0] = 0.5f;
    state.extraFeatures[1] = -0.25f;
    float mean = 2.f;
    float convWeight = convLayer.weights[0][0][0][0];
    float convBias = convLayer.biases[0];
    vector<float> input{mean * convWeight + convBias};
    input.insert(input.end(), state.extraFeatures.begin(), state.extraFeatures.end());
    auto before = learner.forwardPropagation(state);
    array<float, 4> expected{};
    for (size_t action = 0; action < expected.size(); action++)
    {
        expected[action] = denseLayer.biases[action];
        for (size_t feature = 0; feature < input.size(); feature++)
        {
            expected[action] += input[feature] * denseLayer.weights[feature][action];
        }

        checkNear(before[action], expected[action], "Incorrect raw Q-value from pooled channels and extras");
    }

    int action = static_cast<int>(Action::Down);
    float reward = expected[action] + 1.f;
    Transition experience{state, Action::Down, reward, {}, true};
    learner.remember(experience);
    learner.remember(experience);
    experience.state.board.clear();
    experience.state.extraFeatures.clear();
    experience.reward = 50.f;

    // Identical entries have the same mean gradient as one entry, even when the requested batch is larger.
    float learningRate = 0.1f;
    float tdError = expected[action] - reward;
    float pooledGradient = denseLayer.weights[0][action] * tdError;
    // One cell contributes 12/6 to the pooled weight derivative. All six cells contribute 1/6 to the bias.
    convWeight -= learningRate * pooledGradient * mean;
    convBias -= learningRate * pooledGradient;
    denseLayer.biases[action] -= learningRate * tdError;
    for (size_t feature = 0; feature < input.size(); feature++)
    {
        denseLayer.weights[feature][action] -= learningRate * input[feature] * tdError;
    }

    learner.train(5, learningRate);

    // A zero board isolates the updated convolution bias. The original board checks the weight update as well.
    auto zeroBoard = state;
    zeroBoard.board.front()[1][2] = 0.f;
    for (const auto *probe : {&state, &zeroBoard})
    {
        input[0] = probe->board.front()[1][2] / 6.f * convWeight + convBias;
        auto values = learner.forwardPropagation(*probe);
        for (size_t index = 0; index < values.size(); index++)
        {
            float prediction = denseLayer.biases[index];
            for (size_t feature = 0; feature < input.size(); feature++)
            {
                prediction += input[feature] * denseLayer.weights[feature][index];
            }

            checkNear(values[index], prediction, "Incorrect end-to-end batch update", 1e-6f, 1e-5f);
        }
    }
}  //----- end of testPooledBatchUpdate


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
        {"exploration_and_decay", testExplorationAndDecay},
        {"terminal_targets", testTerminalTargets},
        {"empty_next_action_mask", testEmptyNextActionMask},
        {"legal_targets_and_synchronization", testLegalTargetsAndSynchronization},
        {"pooled_batch_update", testPooledBatchUpdate},
        {"learning_from_episode", testLearningFromEpisode},
        {"seeded_training", testSeededTraining}
    };
}  //----- end of getTests
