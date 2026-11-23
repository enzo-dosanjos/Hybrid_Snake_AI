/*************************************************************************
ResultValidationTests - Match evaluation and greedy policy integration
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//- Implementation of the module <ResultValidationTests> (file ResultValidationTests.cpp) -

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <functional>

using namespace std;

//------------------------------------------------------- Personal Include
#include "NeuralAgentTest.h"
#include "src/Agent/DQNAgent.h"
#include "src/DQN/ResultValidation.h"

//------------------------------------------------------------------ Types
class FirstLegalAgent : public Agent {
    public:
        optional<Action> selectAction(const GameEngine &engine) override
        // Algorithm : Record the scheduled seat and return its first legal move.
        {
            int player = engine.getState().currentPlayer;
            seats.push_back(player);
            ActionMask mask = engine.getActionMask(player);
            for (size_t action = 0; action < mask.size(); action++)
            {
                if (mask[action])
                {
                    return static_cast<Action>(action);
                }
            }

            throw logic_error("Called a policy without legal moves");
        }  //----- end of selectAction

        vector<int> seats;
};


//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
void testSeatRotationAndScore()
// Algorithm : Swap the evaluated seat on a board where the first mover wins and check aggregate metrics.
{
    FirstLegalAgent evaluated;
    FirstLegalAgent opponent;
    ResultValidation validation;
    vector<Coord> origins{{}, {0, 0}, {2, 0}};
    auto result = validation.validate(evaluated, {
        {{3, 1, 1, 2, 1}, origins, {ref(opponent)}},
        {{3, 1, 1, 2, 2}, origins, {ref(opponent)}}
    });

    check(result.matches == 2 && result.wins == 1, "The first seat should win both matches");
    check(evaluated.seats == vector<int>{1} && opponent.seats == vector<int>{1},
          "Policies were called for an unexpected seat");
    checkNear(result.meanSurvivalTurns, 2.f, "Both matches end on player two's trapped turn");
    checkNear(validation.calculateScore(result), 0.5f, "Validation score must be the win rate");
}  //----- end of testSeatRotationAndScore


void testMultipleOpponentsAndDeadSlots()
// Algorithm : Run a line-board match where the first opponent is trapped early and its next slot is skipped.
{
    FirstLegalAgent evaluated;
    FirstLegalAgent firstOpponent;
    FirstLegalAgent lastOpponent;
    ResultValidation validation;
    auto result = validation.validate(evaluated, {
        {{6, 1, 1, 3, 2}, {{}, {0, 0}, {1, 0}, {5, 0}}, {ref(firstOpponent), ref(lastOpponent)}}
    });

    check(result.wins == 1, "The middle player should survive");
    check(firstOpponent.seats.empty(), "Trapped and dead players must not be asked for actions");
    check(evaluated.seats == vector<int>{2, 2} && lastOpponent.seats == vector<int>{3},
          "The validator did not dispatch moves to the scheduled policies");
    checkNear(result.meanSurvivalTurns, 5.f, "Count both trapped turns and omit the dead slot");
}  //----- end of testMultipleOpponentsAndDeadSlots


void testGreedyEvaluationPreservesExploration()
// Algorithm : Evaluate at full epsilon, then compare future exploration to a learner that never played evaluation games.
{
    auto config = makeAgentModelConfig();
    DQN learner(config, mt19937(42));
    DQN control(config, mt19937(42));
    DQNAgent evaluated(learner, false);
    FirstLegalAgent opponent;
    ResultValidation validation;

    auto result = validation.validate(evaluated, {
        {{4, 4, 1, 2, 1}, {{}, {0, 2}, {3, 1}}, {ref(opponent)}},
        {{4, 4, 1, 2, 2}, {{}, {3, 1}, {0, 2}}, {ref(opponent)}}
    });
    check(result.matches == 2 && result.meanSurvivalTurns > 2.0, "Evaluation did not play both matches");

    GameEngine engine = makeAgentDecision();
    auto state = referenceDecision(engine, 2, config.features);
    for (int attempt = 0; attempt < 64; attempt++)
    {
        check(learner.selectAction(state, true) == control.selectAction(state, true),
              "Validation advanced the exploration generator");
    }
}  //----- end of testGreedyEvaluationPreservesExploration


void testEmptySetAndMissingOpponent()
// Algorithm : Check the empty result and reject a match without a policy for each other seat.
{
    FirstLegalAgent agent;
    ResultValidation validation;
    auto empty = validation.validate(agent, {});
    check(empty.matches == 0 && empty.wins == 0 && empty.meanSurvivalTurns == 0.0 &&
          validation.calculateScore(empty) == 0.f, "Expected zero metrics for an empty validation set");

    bool rejected = false;
    try
    {
        validation.validate(agent, {{{3, 1, 1, 2, 1}, {{}, {0, 0}, {2, 0}}, {}}});
    }
    catch (const invalid_argument &)
    {
        rejected = true;
    }

    check(rejected && agent.seats.empty(), "An incomplete lineup must fail before starting the match");
}  //----- end of testEmptySetAndMissingOpponent


vector<TestCase> getTests()
// Algorithm : Register this component's tests for the common runner.
{
    return {
        {"seat_rotation_and_score", testSeatRotationAndScore},
        {"multiple_opponents_and_dead_slots", testMultipleOpponentsAndDeadSlots},
        {"greedy_evaluation_preserves_exploration", testGreedyEvaluationPreservesExploration},
        {"empty_set_and_missing_opponent", testEmptySetAndMissingOpponent}
    };
}  //----- end of getTests
