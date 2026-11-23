/*************************************************************************
DQNAgentTests - Scheduled perspective, legality and learner sharing
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//--- Implementation of the module <DQNAgentTests> (file DQNAgentTests.cpp) ---

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <random>

using namespace std;

//------------------------------------------------------- Personal Include
#include "NeuralAgentTest.h"
#include "src/Agent/DQNAgent.h"

//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
void testScheduledPerspective()
// Algorithm : Compare the agent with direct inference for player two where the wrong perspective and illegal maximum differ.
{
    auto engine = makeAgentDecision();
    auto before = engine.getState();
    auto config = makeAgentModelConfig();
    DQN learner(config, mt19937(6));
    DQNAgent agent(learner, false);
    auto expected = referenceDecision(engine, 2, config.features);
    auto values = learner.forwardPropagation(expected);

    auto wrongPerspective = referenceDecision(engine, 1, config.features);
    wrongPerspective.legalActions = expected.legalActions;
    check(learner.selectAction(wrongPerspective, false) == Action::Down, "Wrong perspective must favor Down");
    check(values[2] > values[1] && !expected.legalActions[2], "Fixture must favor an illegal Right");
    check(learner.selectAction(expected, false) == Action::Up, "Scheduled perspective must favor legal Up");
    check(agent.selectAction(engine) == Action::Up, "Agent did not use the scheduled player's encoded decision");
    checkState(engine.getState(), before);
}  //----- end of testScheduledPerspective


void testBorrowedLearner()
// Algorithm : Train the learner after building the agent and verify that the same agent sees the changed ranking.
{
    auto engine = makeAgentDecision();
    auto config = makeAgentModelConfig();
    DQN learner(config, mt19937(6));
    DQNAgent agent(learner, false);
    auto state = referenceDecision(engine, 2, config.features);

    check(agent.selectAction(engine) == Action::Up, "Initial greedy action should be Up");
    learner.remember({state, Action::Down, 5.f, {}, true});
    learner.train(1, 0.04f);
    check(learner.selectAction(state, false) == Action::Down, "Training did not change the reference ranking");
    check(agent.selectAction(engine) == Action::Down, "Agent missed the borrowed learner's update");
}  //----- end of testBorrowedLearner


void testExplorationMode()
// Algorithm : Compare seeded training selections with a reference learner while evaluation remains greedy and preserves the RNG stream.
{
    auto engine = makeAgentDecision();
    auto before = engine.getState();
    auto config = makeAgentModelConfig();
    auto state = referenceDecision(engine, 2, config.features);
    DQN learner(config, mt19937(6));
    DQN reference(config, mt19937(6));
    DQNAgent training(learner, true);
    DQNAgent evaluation(learner, false);
    ActionMask explored;

    checkNoAgentDecision(training);
    checkNoAgentDecision(evaluation);
    for (int attempt = 0; attempt < 32; attempt++)
    {
        check(evaluation.selectAction(engine) == Action::Up, "DQN evaluation explored at full epsilon");
        auto action = training.selectAction(engine);
        check(action && action == reference.selectAction(state, true), "DQN training did not use the learner's policy stream");
        explored.set(static_cast<int>(*action));
    }

    check(explored == state.legalActions, "DQN exploration failed to cover exactly the legal actions");
    checkState(engine.getState(), before);
}  //----- end of testExplorationMode


vector<TestCase> getTests()
// Algorithm : Register the permanent neural-agent behavior cases.
{
    return {
        {"scheduled-perspective", testScheduledPerspective},
        {"borrowed-learner", testBorrowedLearner},
        {"exploration-mode", testExplorationMode}
    };
}  //----- end of getTests
