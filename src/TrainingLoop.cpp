/*************************************************************************
TrainingLoop - Plays learner decisions and their intervening turns
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//--- Implementation of the module <TrainingLoop> (file TrainingLoop.cpp) ---

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <stdexcept>
#include <utility>

using namespace std;

//------------------------------------------------------- Personal Include
#include "TrainingLoop.h"
#include "GameEngine/DecisionEncoding.h"

//---------------------------------------------------------------- PRIVATE
static Action playAction(GameEngine &engine, Agent &agent, ActionMask legalActions)
// Algorithm : Require a legal policy decision and return the action applied by the engine.
{
    auto action = agent.selectAction(engine);

    if (!action || static_cast<size_t>(*action) >= legalActions.size() ||
        !legalActions[static_cast<size_t>(*action)])
    {
        throw logic_error("Training agent did not select a legal action");
    }

    engine.updateStep(*action);
    return *action;
}  //----- end of playAction


static void playOpponentTurns(GameEngine &engine, const vector<reference_wrapper<Agent>> &opponents)
// Algorithm : Process opponent slots through the next learner turn or the end of the game.
{
    int learner = engine.getP();
    while (!engine.isTerminalFor(learner) && engine.getState().currentPlayer != learner)
    {
        int opponent = engine.getState().currentPlayer;
        ActionMask legalActions = engine.getActionMask(opponent);

        if (legalActions.none())
        {
            engine.playerDied(opponent);
            engine.finishTurn();
            continue;
        }

        // The lineup omits the learner's seat.
        int index = opponent - 1 - (opponent > learner);
        playAction(engine, opponents[index].get(), legalActions);
    }
}  //----- end of playOpponentTurns


//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
optional<Transition> playLearnerDecision(
    GameEngine &engine, Agent &behavior, const vector<reference_wrapper<Agent>> &opponents,
    const Reward &reward, const FeatureSelection &features
)
// Algorithm : Capture the snapshots around an executed learner move and all following opponent turns.
{
    if (opponents.size() != static_cast<size_t>(engine.getN() - 1))
    {
        throw invalid_argument("Training requires an opponent for each other seat");
    }

    playOpponentTurns(engine, opponents);
    int learner = engine.getP();
    if (engine.isTerminalFor(learner) || engine.getActionMask(learner).none())
    {
        return nullopt;
    }

    GameState previous = engine.getState();
    EncodedState state = encodeDecision(engine, learner, features);
    Action action = playAction(engine, behavior, state.legalActions);
    playOpponentTurns(engine, opponents);

    EncodedState next = encodeDecision(engine, learner, features);
    bool done = engine.isTerminalFor(learner) || next.legalActions.none();
    float score = reward.computeReward(previous, engine.getState(), learner);
    return Transition{move(state), action, score, move(next), done};
}  //----- end of playLearnerDecision
