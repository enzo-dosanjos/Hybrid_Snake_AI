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
// Algorithm : Tests entropy properties using empty, repeated, balanced and duplicated action histories.
{
    StateAnalyzer analyzer;
    deque<Action> repeated(12, Action::Left);
    deque<Action> balanced{Action::Left, Action::Up, Action::Right, Action::Down};
    deque<Action> doubled = balanced;
    doubled.insert(doubled.end(), balanced.begin(), balanced.end());

    checkNear(analyzer.computeActionEntropy({}), 0.f, "Empty history entropy");
    checkNear(analyzer.computeActionEntropy(repeated), 0.f, "Repeated action entropy");
    check(analyzer.computeActionEntropy(balanced) > 0.f, "Balanced actions need positive entropy");
    checkNear(analyzer.computeActionEntropy(balanced), analyzer.computeActionEntropy(doubled),
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


vector<TestCase> getTests()
// Algorithm : Registers this component’s named test functions for independent selection and execution.
{
    return {
        {"feature_order", testFeatureOrder},
        {"dead_player_slots", testDeadPlayerSlots},
        {"action_entropy", testActionEntropy},
        {"degenerate_boards", testDegenerateBoards}
    };
}  //----- end of getTests
