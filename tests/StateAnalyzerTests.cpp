/*************************************************************************
StateAnalyzerTests - Feature stability and behavioral metrics
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//- Implementation of the module <StateAnalyzerTests> (file StateAnalyzerTests.cpp) -

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <algorithm>

using namespace std;

//------------------------------------------------------- Personal Include
#include "GameTest.h"
#include "src/GameEngine/StateAnalyzer.h"


//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
void testFeatureOrder()
// Algorithm : Tests deterministic feature ordering by comparing every permutation of player insertion order.
{
    GameEngine engine({8, 5, 3, 3, 1}, {{0, 0}, {1, 1}, {6, 3}, {3, 4}});
    StateAnalyzer analyzer;
    auto state = engine.getState();
    auto expected = analyzer.computeExtraFeatures(state, 1);
    vector<int> order{1, 2, 3};

    check(!expected.empty(), "Feature vector is empty");

    do
    {
        auto reordered = state;
        reordered.players.clear();

        for (int player : order)
        {
            reordered.players.emplace(player, state.players.at(player));
        }

        check(analyzer.computeExtraFeatures(reordered, 1) == expected,
              "Feature order depends on player insertion order");
    } while (next_permutation(order.begin(), order.end()));
}  //----- end of testFeatureOrder


void testDeadPlayerSlots()
// Algorithm : Tests fixed feature size and zeroed dead-player metrics by killing and repositioning an opponent.
{
    GameEngine engine({8, 5, 3, 3, 1}, {{0, 0}, {1, 1}, {6, 3}, {3, 4}});
    StateAnalyzer analyzer;
    auto before = analyzer.computeExtraFeatures(engine.getState(), 1);
    engine.playerDied(2);
    auto after = analyzer.computeExtraFeatures(engine.getState(), 1);

    check(before.size() == after.size(), "Death changed input dimensions");
    check(before != after, "Alive status is missing from features");

    auto movedDeadPlayer = engine.getState();
    movedDeadPlayer.players.at(2).snake = {{0, 4}, {0, 3}};

    check(after == analyzer.computeExtraFeatures(movedDeadPlayer, 1),
          "Dead player's geometric metrics were not zeroed");

    for (float value : after)
    {
        check(isfinite(value), "Death produced a non-finite feature");
    }
}  //----- end of testDeadPlayerSlots


void testActionEntropy()
// Algorithm : Tests public entropy features using empty, repeated, balanced and duplicated action histories.
{
    StateAnalyzer analyzer;
    GameEngine engine({5, 5, 3, 2, 1}, {{0, 0}, {0, 0}, {4, 4}});
    auto state = engine.getState();
    deque<Action> repeated(12, Action::Left);
    deque<Action> balanced{Action::Left, Action::Up, Action::Right, Action::Down};
    deque<Action> doubled = balanced;
    doubled.insert(doubled.end(), balanced.begin(), balanced.end());

    // Gameplay slot 1 contains the perspective player's normalized entropy after tanh.
    state.lastMoves[1].clear();
    checkNear(analyzer.computeGameplayMetrics(state, 1)[1], 0.f, "Empty history entropy");
    state.lastMoves[1] = repeated;
    checkNear(analyzer.computeGameplayMetrics(state, 1)[1], 0.f, "Repeated action entropy");
    state.lastMoves[1] = balanced;
    float balancedEntropy = analyzer.computeGameplayMetrics(state, 1)[1];
    checkNear(balancedEntropy, tanh(1.f), "Balanced actions need maximum entropy");
    state.lastMoves[1] = doubled;
    checkNear(balancedEntropy, analyzer.computeGameplayMetrics(state, 1)[1],
              "Entropy depends on sample count instead of distribution");
}  //----- end of testActionEntropy


void testDegenerateBoards()
// Algorithm : Tests numerical validity by checking every feature on narrow and rectangular boards.
{
    StateAnalyzer analyzer;

    for (auto size : vector<Coord>{{1, 4}, {4, 1}, {3, 7}, {7, 3}})
    {
        GameEngine engine({size.x, size.y, 1, 2, 1},
                          {{0, 0}, {0, 0}, {size.x - 1, size.y - 1}});

        for (float value : analyzer.computeExtraFeatures(engine.getState(), 1))
        {
            check(isfinite(value), "Board dimensions produced a non-finite feature");
        }
    }
}  //----- end of testDegenerateBoards


void testFeatureValues()
// Algorithm : Tests every schema slot against hand-calculated distances, growth, turn and history values
{
    GameEngine engine({3, 4, 3, 2, 1}, {{0, 0}, {0, 0}, {2, 0}});
    auto state = engine.getState();
    state.players.at(1).snake = {{0, 0}, {0, 1}};
    state.players.at(2).snake = {{2, 0}, {2, 1}, {2, 2}};
    state.round = 1;
    state.currentPlayer = 2;
    state.lastMoves[1] = {Action::Left, Action::Up};
    state.lastMoves[2] = {Action::Left, Action::Up, Action::Right, Action::Down};
    GameEngine position(state);
    StateAnalyzer analyzer;

    // The legacy distance scale is sqrt(W*W + H*H), which is five on this board
    float diagonal = sqrt(5.f) / 5.f;
    float farthest = sqrt(8.f) / 5.f;
    vector<float> expected = {
        1.f, 0.4f, farthest, (0.4f + diagonal + farthest) / 3.f,
        0.4f, diagonal, (0.4f + diagonal) / 2.f,
        0.3f, 2.f / 3.f, 1.f / 6.f, 5.f / 48.f,
        1.f, tanh(0.5f), 1.f, 0.25f, 1.f, tanh(1.f)
    };
    auto actual = analyzer.computeExtraFeatures(position.getState(), 1);
    check(actual.size() == expected.size(), "Unexpected schema dimensions");

    for (size_t feature = 0; feature < expected.size(); feature++)
    {
        checkNear(actual[feature], expected[feature], "Wrong feature at index " + to_string(feature));
    }

    check(analyzer.computeExtraFeatures(position.getState(), 2) != actual,
          "Feature perspective was ignored");
    check(EXTRA_FEATURE_SCHEMA_VERSION == 1, "Feature schema changed without updating its numerical contract");
}  //----- end of testFeatureValues


void testClockAndHistoryContext()
// Algorithm : Tests fixed dimensions, scheduled-player sensitivity and independent opponent histories
{
    StateAnalyzer analyzer;

    for (int players : {1, 2, 3})
    {
        GameEngine engine({5, 5, 3, players, 1}, {{0, 0}, {0, 0}, {4, 4}, {4, 0}});
        check(analyzer.computeExtraFeatures(engine.getState(), 1).size() ==
              static_cast<size_t>(StateAnalyzer::getFeatureCount(players)), "Wrong fixed feature count");
    }

    GameEngine engine({5, 5, 3, 3, 1}, {{0, 0}, {0, 0}, {4, 4}, {4, 0}});
    auto state = engine.getState();
    auto before = analyzer.computeExtraFeatures(state, 1);
    state.currentPlayer = 2;
    check(analyzer.computeExtraFeatures(state, 1) != before, "Scheduled player is missing");
    state.currentPlayer = 1;
    state.lastMoves[2] = {Action::Left, Action::Up};
    check(analyzer.computeExtraFeatures(state, 1) != before, "Opponent history is missing");
    state.players.at(2).playerInfo.alive = false;
    auto dead = analyzer.computeExtraFeatures(state, 1);
    state.lastMoves[2] = {Action::Left, Action::Up, Action::Right, Action::Down};
    check(analyzer.computeExtraFeatures(state, 1) == dead, "Dead opponent history was not zeroed");

    for (int round = 0; round < 6; round++)
    {
        state.round = round;
        checkNear(analyzer.computeGrowthSizeMetrics(state, 1)[1],
                  vector<float>{0.f, 2.f / 3.f, 1.f / 3.f}[round % 3], "Growth phase follows the wrong clock");
    }

    // Gameplay slot 4 holds the first opponent's wall proximity; restore its alive flag.
    state.players.at(2).playerInfo.alive = true;
    state.players.at(2).snake = {{0, 0}};
    float cornerProximity = analyzer.computeGameplayMetrics(state, 1)[4];
    checkNear(cornerProximity, 1.f, "Corner proximity should be maximal");
    state.players.at(2).snake = {{4, 4}};
    checkNear(cornerProximity, analyzer.computeGameplayMetrics(state, 1)[4], "Opposite corners differ");
}  //----- end of testClockAndHistoryContext


vector<TestCase> getTests()
// Algorithm : Registers this component’s named test functions for independent selection and execution.
{
    return {
        {"feature_order", testFeatureOrder},
        {"feature_values", testFeatureValues},
        {"clock_and_history_context", testClockAndHistoryContext},
        {"dead_player_slots", testDeadPlayerSlots},
        {"action_entropy", testActionEntropy},
        {"degenerate_boards", testDegenerateBoards}
    };
}  //----- end of getTests
