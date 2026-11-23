/*************************************************************************
HybridAgentTests - Legal normalization, weighting and exploration
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//- Implementation of the module <HybridAgentTests> (file HybridAgentTests.cpp) -

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <array>
#include <limits>
#include <random>
#include <stdexcept>

using namespace std;

//------------------------------------------------------- Personal Include
#include "NeuralAgentTest.h"
#include "src/Agent/HybridAgent.h"

//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
void testNormalizedCompromise()
// Algorithm : Check both policy endpoints and a compromise that requires scaling over legal moves only.
{
    auto engine = makeAgentDecision();
    auto before = engine.getState();
    auto config = makeAgentModelConfig();
    DQN learner(config, mt19937(6));
    MinimaxParams params;
    params.depth = 1;
    auto search = Minimax(params).calculateMoveScores(engine);
    auto state = referenceDecision(engine, 2, config.features);
    auto q = learner.forwardPropagation(state);

    check(search.size() == 3 && !state.legalActions[2], "Fixture must exclude Right");
    check(search.at(Action::Left) > search.at(Action::Down) && search.at(Action::Down) > search.at(Action::Up),
          "Search must rank Left, Down, Up");
    check(q[2] > q[1] && q[1] > q[3] && q[3] > q[0], "Q-values must rank Right, Up, Down, Left");

    // At equal weights Left and Up each receive one unit before weighting.
    // Down beats them only when its two fractions sum to more than one.
    double searchFraction = (search.at(Action::Down) - search.at(Action::Up)) /
                            (search.at(Action::Left) - search.at(Action::Up));
    double legalQFraction = (q[3] - q[0]) / (q[1] - q[0]);
    double unmaskedQFraction = (q[3] - q[0]) / (q[2] - q[0]);
    check(searchFraction + legalQFraction > 1.0, "Legal normalization must favor the compromise");
    check(searchFraction + unmaskedQFraction < 1.0, "Including illegal Right must change the result");
    check(search.at(Action::Left) + q[0] > search.at(Action::Down) + q[3],
          "Raw addition must miss the compromise");

    HybridAgent searchOnly(learner, params, 0.f, false);
    HybridAgent hybrid(learner, params, 0.5f, false);
    HybridAgent neuralOnly(learner, params, 1.f, false);
    check(searchOnly.selectAction(engine) == Action::Left, "Lambda zero did not select the search maximum");
    check(hybrid.selectAction(engine) == Action::Down, "Hybrid did not select the normalized compromise");
    check(neuralOnly.selectAction(engine) == Action::Up, "Lambda one did not select the legal Q maximum");
    check(learner.forwardPropagation(state) == q, "Hybrid selection changed the raw Q-values");
    checkState(engine.getState(), before);
}  //----- end of testNormalizedCompromise


void testEqualScoresAndTies()
// Algorithm : Check that a constant scorer does not affect ranking and that a combined tie follows enum order.
{
    auto engine = makeAgentDecision();
    auto config = makeAgentModelConfig();
    DQN learner(config, mt19937(6));
    MinimaxParams normal;
    normal.depth = 1;
    MinimaxParams flat{{0, 0, 0, 0, 0, 0, 0}, 1};
    HybridAgent flatSearch(learner, flat, 0.5f, false);
    check(flatSearch.selectAction(engine) == Action::Up, "Equal search scores changed the Q ranking");

    // The single hidden unit is inactive for this seeded network and input, giving a zero action head.
    config.denseHiddenSizes = {1};
    DQN flatLearner(config, mt19937(1));
    auto state = referenceDecision(engine, 2, config.features);
    check(flatLearner.forwardPropagation(state) == array<float, 4>{}, "Fixture must have equal Q-values");
    HybridAgent flatQ(flatLearner, normal, 0.5f, false);
    HybridAgent bothFlat(flatLearner, flat, 0.5f, false);
    check(flatQ.selectAction(engine) == Action::Left, "Equal Q-values changed the search ranking");
    check(bothFlat.selectAction(engine) == Action::Left, "Equal combined scores did not use enum order");

    GameEngine corner({4, 4, 2, 2, 1}, {{0, 0}, {1, 1}, {3, 0}});
    corner.finishTurn();
    auto cornerState = referenceDecision(corner, 2, config.features);
    auto cornerQ = learner.forwardPropagation(cornerState);
    check(cornerState.legalActions == ActionMask(9) && cornerQ[3] > cornerQ[0],
          "Corner fixture must have exactly two legal moves and favor Down in Q");

    const auto &weights = normal.rewards;
    MinimaxParams reversed{{-weights.selfSpace, -weights.oppSpace, -weights.selfBlockRisk,
                            -weights.oppBlockRisk, -weights.selfHeatRisk, -weights.oppHeatRisk,
                            -weights.snakeLength}, 1};
    auto scores = Minimax(reversed).calculateMoveScores(corner);
    check(scores.at(Action::Left) > scores.at(Action::Down), "Corner search must favor Left");
    HybridAgent opposed(learner, reversed, 0.5f, false);
    check(opposed.selectAction(corner) == Action::Left, "Opposite best scores did not tie in enum order");
}  //----- end of testEqualScoresAndTies


void testSharedExploration()
// Algorithm : Follow the seeded exploration stream, decay epsilon, then verify that exploitation sees a training update.
{
    auto engine = makeAgentDecision();
    auto before = engine.getState();
    auto config = makeAgentModelConfig();
    auto state = referenceDecision(engine, 2, config.features);
    DQN learner(config, mt19937(6));
    DQN reference(config, mt19937(6));
    MinimaxParams params;
    params.depth = 1;
    HybridAgent training(learner, params, 0.5f, true);
    HybridAgent evaluation(learner, params, 0.5f, false);
    ActionMask explored;

    checkNoAgentDecision(training);
    checkNoAgentDecision(evaluation);
    for (int attempt = 0; attempt < 32; attempt++)
    {
        check(evaluation.selectAction(engine) == Action::Down, "Hybrid evaluation explored at full epsilon");
        auto action = training.selectAction(engine);
        check(action && action == reference.selectAction(state, true), "Hybrid did not use the learner's exploration stream");
        explored.set(static_cast<int>(*action));
    }

    check(explored == state.legalActions, "Hybrid exploration failed to cover exactly the legal actions");

    learner.remember({state, Action::Left, 5.f, {}, true});
    learner.train(1, 0.f);
    for (int attempt = 0; attempt < 16; attempt++)
    {
        check(training.selectAction(engine) == Action::Down, "Hybrid did not exploit after the shared epsilon decayed");
    }

    learner.train(1, 0.04f);
    check(learner.selectAction(state, false) == Action::Left, "Training did not change the legal Q maximum");
    check(training.selectAction(engine) == Action::Left, "Hybrid missed the borrowed learner's update");
    checkState(engine.getState(), before);
}  //----- end of testSharedExploration


void testInvalidLambda()
// Algorithm : Attempt construction with out-of-range and unordered weights and require rejection.
{
    DQN learner(makeAgentModelConfig(), mt19937(6));
    for (float lambda : {-0.1f, 1.1f, numeric_limits<float>::quiet_NaN()})
    {
        bool rejected = false;
        try
        {
            HybridAgent agent(learner, MinimaxParams{}, lambda, false);
        }
        catch (const invalid_argument &)
        {
            rejected = true;
        }

        check(rejected, "Hybrid accepted a lambda outside [0, 1]");
    }
}  //----- end of testInvalidLambda


vector<TestCase> getTests()
// Algorithm : Register the permanent hybrid-agent behavior cases.
{
    return {
        {"normalized-compromise", testNormalizedCompromise},
        {"equal-scores-and-ties", testEqualScoresAndTies},
        {"shared-exploration", testSharedExploration},
        {"invalid-lambda", testInvalidLambda}
    };
}  //----- end of getTests
