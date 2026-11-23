/*************************************************************************
RewardTests - Decision interval outcomes and shaping rewards
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//--------- Implementation of the module <RewardTests> (file RewardTests.cpp) ---------

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <cmath>

using namespace std;

//------------------------------------------------------- Personal Include
#include "Test.h"
#include "src/DQN/Reward.h"
#include "src/GameEngine/GameEngine.h"

//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
void testObservedElimination()
// Algorithm : Drive an opponent collision between learner turns and verify that its reward occurs once.
{
    const GameConfig config{8, 8, 1, 3, 1};
    GameEngine game(config, {{0, 0}, {3, 3}, {0, 0}, {7, 7}});
    Reward reward(config);
    GameState previous = game.getState();

    game.updateStep(Action::Right);
    game.updateStep(Action::Up);
    game.updateStep(Action::Left);

    // Three elapsed slots, two learner segments and one surviving opponent at corner proximity 1.
    // The surviving opponent still has two moves, so it earns neither mobility term.
    float expected = 128.f * (1.f + 3.f/64.f) + 1.6f + 48.f/(1.f + exp(-2.8f));
    checkNear(reward.computeReward(previous, game.getState(), 1), expected,
              "Opponent collision during the interval was scored incorrectly");

    previous = game.getState();
    game.updateStep(Action::Down);
    game.finishTurn();
    game.updateStep(Action::Up);

    // The dead slot advances the schedule. The living opponent has more exits and proximity 7/8.
    expected = 2.4f + 48.f/(1.f + exp(-2.3f));
    checkNear(reward.computeReward(previous, game.getState(), 1), expected,
              "An earlier opponent death was scored again");
}  //----- end of testObservedElimination


void testTerminalWin()
// Algorithm : Check that the final observed elimination earns exactly one scaled win.
{
    const GameConfig config{8, 8, 2, 2, 1};
    GameEngine game(config, {{0, 0}, {3, 3}, {0, 0}});
    Reward reward(config);
    GameState previous = game.getState();

    game.updateStep(Action::Right);
    game.updateStep(Action::Up);

    // The second scheduled slot ends the game. The M=2 turn budget is 0.8 times 128.
    float expected = 128.f * (1.f + 2.f/102.4f);
    checkNear(reward.computeReward(previous, game.getState(), 1), expected,
              "Final elimination earned more or less than one win");
}  //----- end of testTerminalWin


void testLearnerDeathInMultiplayer()
// Algorithm : End a learner's interval with its collision while two opponents remain alive.
{
    const GameConfig config{8, 8, 2, 3, 2};
    GameEngine game(config, {{0, 0}, {3, 3}, {0, 0}, {7, 7}});
    Reward reward(config);
    game.updateStep(Action::Right);
    GameState previous = game.getState();

    game.updateStep(Action::Up);

    // The learner was scheduled second. After its collision, the game is at slot two of round zero.
    float progress = 2.f/102.4f;
    float expected = -128.f * (0.2f + 0.8f * progress * progress);
    check(!game.isTerminalState(), "Fixture must leave a multiplayer game in progress");
    checkNear(reward.computeReward(previous, game.getState(), 2), expected,
              "Learner death did not end its reward interval with a loss");
}  //----- end of testLearnerDeathInMultiplayer


void testSetupAndEscapeReduction()
// Algorithm : Verify setup is paid on entry while survival and wall pressure are paid on continuing intervals.
{
    const GameConfig config{8, 8, 1, 2, 1};
    GameEngine game(config, {{0, 0}, {3, 3}, {2, 1}});
    Reward reward(config);
    GameState previous = game.getState();

    game.updateStep(Action::Right);
    game.updateStep(Action::Up);

    // On the wall, the opponent loses the upward and backward moves and has proximity 3/4.
    float wallReward = 48.f/(1.f + exp(-1.8f));
    float expected = 1.6f + 120.f + wallReward + 19.2f * (2.f/4.f);
    checkNear(reward.computeReward(previous, game.getState(), 1), expected,
              "First setup or escape-route reduction was scored incorrectly");

    previous = game.getState();
    game.updateStep(Action::Down);
    game.updateStep(Action::Right);

    // The opponent remains on the wall with two exits and the same proximity.
    expected = 2.4f + wallReward;
    checkNear(reward.computeReward(previous, game.getState(), 1), expected,
              "An opponent that remained constrained earned another setup bonus");

    GameState withHistory = game.getState();
    withHistory.lastMoves[2] = {Action::Right, Action::Right, Action::Right};
    checkNear(reward.computeReward(previous, withHistory, 1), expected,
              "Reward depended on diagnostic action history");

    game.updateStep(Action::Right);
    game.updateStep(Action::Down);
    game.updateStep(Action::Down);
    game.updateStep(Action::Right);
    previous = game.getState();

    game.updateStep(Action::Left);
    game.updateStep(Action::Up);

    // The opponent had reopened three moves, then returned to the wall with only one exit.
    expected = 4.8f + 120.f + wallReward + 19.2f * (2.f/4.f);
    checkNear(reward.computeReward(previous, game.getState(), 1), expected,
              "A second entry into a constrained position did not earn setup");
}  //----- end of testSetupAndEscapeReduction


void testTrappedPlayers()
// Algorithm : Use actual engine masks to distinguish a trapped learner from a living, trapped opponent.
{
    const GameConfig config{2, 2, 1, 2, 1};
    GameEngine game(config, {{0, 0}, {0, 0}, {1, 1}});
    Reward reward(config);
    GameState previous = game.getState();

    game.updateStep(Action::Right);
    game.updateStep(Action::Left);

    // Both snakes retain their tails after growing. The learner's next turn has no legal move.
    check(game.isAlive(1) && game.getActionMask(1).none(), "Fixture must leave the learner alive but trapped");
    float expected = -8.f * (0.2f + 0.8f * 0.5f * 0.5f);
    checkNear(reward.computeReward(previous, game.getState(), 1), expected,
              "The learner's empty next legal mask did not produce a loss");

    const GameConfig largerConfig{4, 4, 1, 2, 2};
    GameState setup = GameEngine(largerConfig, {{0, 0}, {1, 0}, {0, 2}}).getState();
    setup.currentPlayer = 2;
    setup.players.at(1).snake = {{1, 0}, {2, 0}};
    setup.players.at(2).snake = {{0, 2}, {0, 3}, {1, 3}};
    GameEngine largerGame(setup);
    Reward largerReward(largerConfig);
    previous = largerGame.getState();

    largerGame.updateStep(Action::Up);
    largerGame.updateStep(Action::Left);

    // Player one has two exits before the interval and gets trapped in the corner.
    // Player two survives with four segments and can still move right.
    check(largerGame.isAlive(1) && largerGame.getActionMask(1).none() &&
          largerGame.getActionMask(2).any(), "Fixture must trap only player one");
    expected = 1.6f + 24.f/(1.f + exp(-2.8f)) + 9.6f * (2.f/4.f);
    checkNear(largerReward.computeReward(previous, largerGame.getState(), 2), expected,
              "A predicted opponent death earned an elimination reward");
}  //----- end of testTrappedPlayers


vector<TestCase> getTests()
// Algorithm : Register the component's named reward tests for the common runner.
{
    return {
        {"observed_elimination", testObservedElimination},
        {"terminal_win", testTerminalWin},
        {"learner_death_in_multiplayer", testLearnerDeathInMultiplayer},
        {"setup_and_escape_reduction", testSetupAndEscapeReduction},
        {"trapped_players", testTrappedPlayers}
    };
}  //----- end of getTests
