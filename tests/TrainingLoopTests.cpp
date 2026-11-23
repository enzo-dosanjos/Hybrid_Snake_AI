/*************************************************************************
TrainingLoopTests - Checks decision intervals in training games
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//- Implementation of the module <TrainingLoopTests> (file TrainingLoopTests.cpp) -

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <functional>

using namespace std;

//------------------------------------------------------- Personal Include
#include "Test.h"
#include "src/TrainingLoop.h"

//------------------------------------------------------------------ Types
class FixedActionAgent : public Agent {
    public:
        explicit FixedActionAgent(Action p_action) : action(p_action)
        // Algorithm : Store the direction to use on each requested turn.
        {
        }  //----- end of FixedActionAgent


        optional<Action> selectAction(const GameEngine &engine) override
        // Algorithm : Record the requested seat and return the scripted direction.
        {
            seats.push_back(engine.getState().currentPlayer);
            return action;
        }  //----- end of selectAction

        vector<int> seats;

    private:
        Action action;
};


//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
void testOpponentTurnsAndTerminalWin()
// Algorithm : Play two learner decisions on a corridor and check the scheduled snapshots and terminal reward.
{
    GameConfig config{6, 1, 1, 3, 2};
    GameEngine engine(config, {{}, {0, 0}, {1, 0}, {5, 0}});
    FixedActionAgent trapped(Action::Right);
    FixedActionAgent behavior(Action::Right);
    FixedActionAgent opponent(Action::Left);
    vector<reference_wrapper<Agent>> opponents{trapped, opponent};
    Reward reward(config);
    FeatureSelection features{false, false, false, false, false};
    ActionMask right;
    right.set(static_cast<size_t>(Action::Right));

    auto first = playLearnerDecision(engine, behavior, opponents, reward, features);
    check(first && !first->done && first->action == Action::Right,
          "The first executed action must produce a continuing transition");
    check(first->state.legalActions == right && first->nextState.legalActions == right,
          "The learner can only move right at both decisions");
    check(first->state.extraFeatures == vector<float>({6, 1, 1, 0, 0, 2, 2, 2, 1, 1, 0, 3, 1}) &&
          first->nextState.extraFeatures == vector<float>({6, 1, 1, 0, 1, 2, 2, 2, 1, 1, 0, 3, 1}),
          "Replay must encode each scheduled learner decision, including the earlier death");
    check(first->state.board[0][0][1] == 1.f && first->nextState.board[0][0][2] == 1.f &&
          first->state.board[8][0][5] == 1.f && first->nextState.board[8][0][4] == 1.f,
          "The replay interval must contain moves by both the learner and the opponent");
    // Two surviving segments and maximum wall proximity contribute shaping, with no change in escape routes.
    float continuing = sqrt(6.f) * (0.2f + 6.f / (1.f + exp(-2.8f)));
    checkNear(first->reward, continuing, "The prior death must not count as an elimination");

    auto second = playLearnerDecision(engine, behavior, opponents, reward, features);
    check(second && second->done && second->action == Action::Right && engine.isTerminalState(),
          "The second executed decision must end on the opponent's trapped turn");
    check(second->nextState.extraFeatures == vector<float>({6, 1, 1, 0, 2, 1, 2, 2, 1, 1, 0, 3, 0}),
          "The terminal snapshot must retain the observed deaths and final schedule");
    check(trapped.seats.empty() && opponent.seats == vector<int>{3} &&
          behavior.seats == vector<int>({2, 2}) &&
          engine.getState().lastMoves.at(2) == deque<Action>({Action::Right, Action::Right}),
          "Trapped and dead slots must not request actions or duplicate decisions");

    // One win at six elapsed slots on a six-cell board, with no additional elimination bonus.
    checkNear(second->reward, 24.f, "The win reward must use the complete interval");
    check(!playLearnerDecision(engine, behavior, opponents, reward, features),
          "A terminal game must not generate more replay");
}  //----- end of testOpponentTurnsAndTerminalWin


void testEmptyNextMaskEndsEpisode()
// Algorithm : Let an opponent move into the last free cell and check the resulting terminal replay sample.
{
    GameConfig config{4, 1, 1, 2, 1};
    GameEngine engine(config, {{}, {0, 0}, {3, 0}});
    FixedActionAgent behavior(Action::Right);
    FixedActionAgent opponent(Action::Left);
    Reward reward(config);

    auto transition = playLearnerDecision(engine, behavior, {ref(opponent)}, reward, FeatureSelection{});
    check(transition && transition->done && transition->nextState.legalActions.none() &&
          transition->nextState.board[4][0][2] == 1.f,
          "An opponent move can exhaust the learner's actions before the learner dies");
    check(engine.getRound() == 1 && engine.getState().currentPlayer == 1 && engine.isAlive(1),
          "The transition must end at the next learner slot without adding a fabricated turn");

    // Two elapsed slots give half the turn budget, for a loss of -8 * (0.2 + 0.8 * 0.25).
    checkNear(transition->reward, -3.2f, "The empty next mask must produce the snapshot loss");
    check(!playLearnerDecision(engine, behavior, {ref(opponent)}, reward, FeatureSelection{}) &&
          behavior.seats == vector<int>{1},
          "A trapped learner must not be asked for an action");
}  //----- end of testEmptyNextMaskEndsEpisode


void testIllegalBehaviorIsRejected()
// Algorithm : Return a blocked direction and verify that it cannot change the learner's scheduled turn.
{
    GameConfig config{4, 1, 1, 2, 1};
    GameEngine engine(config, {{}, {0, 0}, {3, 0}});
    FixedActionAgent behavior(Action::Left);
    FixedActionAgent opponent(Action::Left);
    Reward reward(config);
    bool rejected = false;

    try
    {
        playLearnerDecision(engine, behavior, {ref(opponent)}, reward, FeatureSelection{});
    }
    catch (const logic_error &)
    {
        rejected = true;
    }

    check(rejected && engine.getState().currentPlayer == 1 && engine.getState().lastMoves.at(1).empty() &&
          engine.isAlive(1) && opponent.seats.empty(), "Illegal actions must fail before applying a move");
}  //----- end of testIllegalBehaviorIsRejected


vector<TestCase> getTests()
// Algorithm : Register the training workflow tests with the component runner.
{
    return {
        {"opponent_turns_and_terminal_win", testOpponentTurnsAndTerminalWin},
        {"empty_next_mask_ends_episode", testEmptyNextMaskEndsEpisode},
        {"illegal_behavior_is_rejected", testIllegalBehaviorIsRejected}
    };
}  //----- end of getTests
