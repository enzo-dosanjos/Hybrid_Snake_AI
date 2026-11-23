/*************************************************************************
SpaceRiskAnalyzerTests - Territory, reach timing and spatial risk checks
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//--- Implementation of the module <SpaceRiskAnalyzerTests> (file SpaceRiskAnalyzerTests.cpp) ---

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <limits>

using namespace std;

//------------------------------------------------------- Personal Include
#include "GameTest.h"
#include "src/Minimax/SpaceRiskAnalyzer.h"

//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
void testIndependentArrivals()
// Algorithm : Tests independent paths and scheduled arrival times against hand-counted routes on a small rectangle.
{
    GameEngine engine({3, 2, 3, 2, 1}, {{0, 0}, {0, 0}, {2, 0}});
    SpaceRiskAnalyzer analyzer;
    auto state = engine.getState();
    auto before = state;
    auto timing = analyzer.computeReachTiming(state);
    checkState(state, before);

    check(timing.slots.size() == 2 && timing.slots[0].size() == 3, "Wrong timing dimensions");
    check(timing.slots[0][1].size() == 2, "Shared cell lost an opponent arrival");
    check(timing.slots[0][1].at(1) == 1 && timing.slots[0][1].at(2) == 2,
          "First moves use the wrong scheduled slots");
    check(timing.slots[1][1].at(1) == 3 && timing.slots[1][1].at(2) == 4,
          "Second moves use the wrong scheduled slots");
    check(timing.slots[0][0].at(1) == 0 && timing.slots[0][0].count(2) == 0,
          "An occupied head was traversed");

    state.currentPlayer = 2;
    auto reversed = analyzer.computeReachTiming(state);
    check(reversed.slots[0][1].at(1) == 2 && reversed.slots[0][1].at(2) == 1,
          "Arrival times ignore the scheduled player");
    checkState(engine.getState(), before);
}  //----- end of testIndependentArrivals


void testContestedTerritory()
// Algorithm : Tests exact territory and blocking pressure while ensuring equal-distance cells remain contested across turn orders.
{
    GameEngine engine({3, 2, 3, 2, 1}, {{0, 0}, {0, 0}, {2, 0}});
    auto state = engine.getState();
    SpaceRiskAnalyzer analyzer;
    auto areas = analyzer.calculateAreas(state);

    // Each player owns its head and the cell below; the middle column is equally distant.
    check(areas.space == vector<float>({0.f, 2.f, 2.f}), "Contested cells were assigned territory");
    // Only the lower owned cell touches its own head: (1 + 0.5) / 2 neighbors / 2 owned cells.
    checkNear(areas.blockRisk[1], 0.375f, "Wrong own-head blocking pressure");
    checkNear(areas.blockRisk[2], 0.375f, "Blocking pressure is asymmetric");

    state.currentPlayer = 2;
    auto reordered = analyzer.calculateAreas(state);
    check(areas.space == reordered.space && areas.blockRisk == reordered.blockRisk,
          "Turn priority broke distance-based territory ties");

    auto first = state.players.at(1);
    state.players.erase(1);
    state.players.emplace(1, first);
    check(analyzer.calculateAreas(state).space == areas.space, "Player insertion order changed territory");
}  //----- end of testContestedTerritory


void testRiskValues()
// Algorithm : Tests heatmap decay and perspective changes against independently calculated scheduled-arrival risks.
{
    GameEngine engine({3, 2, 3, 2, 1}, {{0, 0}, {0, 0}, {2, 0}});
    SpaceRiskAnalyzer analyzer(5, 0.7f);
    auto timing = analyzer.computeReachTiming(engine.getState());
    auto own = analyzer.calculateRiskHeatmap(timing, 1);
    auto opponent = analyzer.calculateRiskHeatmap(timing, 2);

    // At the top middle cell, arrivals are half a round and one round after the snapshot.
    checkNear(own[0][1], 0.7f / 2.f, "Later opponent pressure is wrong");
    checkNear(opponent[0][1], 0.7f / 1.5f, "Earlier opponent pressure is wrong");
    // The lower middle cell is two moves away, with opponent arrival at round-time two.
    checkNear(own[1][1], 0.49f / 3.f, "Second-distance decay is wrong");
    checkNear(own[0][2], 0.f, "An unreachable opponent head has pressure");
    checkNear(analyzer.calculateDirectionalRisk(own, {0, 0}), 0.35f + 0.175f,
              "Directional risk did not sum the two neighbors");

    auto state = engine.getState();
    state.currentPlayer = 2;
    auto reversed = analyzer.analyzeState(state, 1);
    checkNear(reversed.heatmap[0][1], opponent[0][1], "Risk did not follow turn priority");
}  //----- end of testRiskValues


void testDeadBodiesAndSlots()
// Algorithm : Tests retained obstacles and dead schedule slots using a dead snake that separates two living players.
{
    GameEngine engine({5, 3, 3, 3, 1}, {{0, 0}, {0, 1}, {2, 0}, {4, 1}});
    auto state = engine.getState();
    state.players.at(2).snake = {{2, 0}, {2, 1}, {2, 2}};
    state.players.at(2).playerInfo.alive = false;
    SpaceRiskAnalyzer analyzer;
    auto timing = analyzer.computeReachTiming(state);
    auto result = analyzer.analyzeState(state, 1);

    check(result.areas.space == vector<float>({0.f, 6.f, 0.f, 6.f}), "Dead wall did not separate territory");
    checkNear(result.areas.blockRisk[2], 0.f, "Dead player retained blocking metrics");
    check(timing.slots[1][1].at(1) == 1 && timing.slots[1][3].at(3) == 3,
          "Dead slot was removed from the first-move schedule");
    check(timing.slots[0][3].at(3) == 6, "Dead slot was removed from later rounds");

    for (int y = 0; y < 3; y++)
    {
        check(timing.slots[y][2].empty(), "Dead snake emitted or admitted reach paths");
        check(timing.slots[y][3].count(1) == 0, "Reach search crossed a retained dead body");
    }

    // A dead head is an ordinary obstacle, without the extra living-opponent head weight.
    GameEngine narrow({3, 1, 3, 2, 1}, {{0, 0}, {0, 0}, {1, 0}});
    narrow.playerDied(2);
    checkNear(analyzer.calculateAreas(narrow.getState()).blockRisk[1], 1.f, "Dead head kept a living-head bonus");
}  //----- end of testDeadBodiesAndSlots


void testLookaheadAndStaticOccupancy()
// Algorithm : Tests exact reach cutoffs and full-board territory while checking that tails remain fixed during analysis.
{
    GameEngine engine({6, 1, 3, 1, 1}, {{0, 0}, {0, 0}});
    SpaceRiskAnalyzer oneMove(1);
    auto timing = oneMove.computeReachTiming(engine.getState());
    check(timing.slots[0][1].at(1) == 1, "Lookahead excluded its boundary");
    check(timing.slots[0][2].empty(), "Reach extended beyond lookahead");
    checkNear(oneMove.calculateAreas(engine.getState()).space[1], 6.f, "Territory was limited by risk lookahead");

    SpaceRiskAnalyzer zeroMoves(0);
    auto zeroTiming = zeroMoves.computeReachTiming(engine.getState());
    check(zeroTiming.slots[0][0].at(1) == 0 && zeroTiming.slots[0][1].empty(), "Zero horizon expanded a move");

    auto state = engine.getState();
    state.players.at(1).snake = {{0, 0}, {1, 0}};
    state.round = 1;
    checkNear(oneMove.calculateAreas(state).space[1], 1.f, "Static analysis released the tail");
    state.round = 3;
    checkNear(oneMove.calculateAreas(state).space[1], 1.f, "Growth phase changed fixed occupancy");
}  //----- end of testLookaheadAndStaticOccupancy


void testSnapshotIndependence()
// Algorithm : Tests repeated analysis across positions, configurations and perspectives against fresh analyzers without changing inputs.
{
    GameEngine first({4, 3, 3, 2, 1}, {{0, 0}, {0, 0}, {3, 2}});
    GameEngine second({2, 5, 1, 2, 1}, {{0, 0}, {1, 0}, {0, 4}});
    SpaceRiskAnalyzer analyzer;
    auto before = first.getState();
    auto expected = analyzer.analyzeState(before, 1);
    analyzer.analyzeState(second.getState(), 2);
    auto repeated = analyzer.analyzeState(before, 1);

    check(expected.areas.space == repeated.areas.space && expected.areas.blockRisk == repeated.areas.blockRisk &&
          expected.heatmap == repeated.heatmap, "Previous analysis contaminated a snapshot");
    checkState(first.getState(), before);

    auto changed = before;
    changed.players.at(2).snake = {{2, 1}, {3, 1}, {3, 2}};
    auto actual = analyzer.analyzeState(changed, 1);
    auto fresh = SpaceRiskAnalyzer().analyzeState(changed, 1);
    check(actual.areas.space == fresh.areas.space && actual.heatmap == fresh.heatmap, "Geometry changes reused old reach data");
    check(actual.heatmap == analyzer.calculateRiskHeatmap(analyzer.computeReachTiming(changed), 1),
          "Combined analysis differs from bounded heatmap analysis");
    checkState(first.getState(), before);
}  //----- end of testSnapshotIndependence


void testBoardDimensions()
// Algorithm : Tests single-cell and rectangular boards using exact single-player area counts and orthogonal heatmap sums.
{
    SpaceRiskAnalyzer analyzer;

    for (Coord size : vector<Coord>{{1, 1}, {1, 5}, {5, 1}, {3, 7}, {7, 3}})
    {
        GameEngine engine({size.x, size.y, 3, 1, 1}, {{0, 0}, {0, 0}});
        auto result = analyzer.analyzeState(engine.getState(), 1);
        check(result.heatmap.size() == static_cast<size_t>(size.y) &&
              result.heatmap[0].size() == static_cast<size_t>(size.x), "Heatmap dimensions are transposed");
        checkNear(result.areas.space[1], float(size.x * size.y), "Single player did not reach every cell");
        check(isfinite(result.areas.blockRisk[1]), "Blocking pressure is not finite");
        checkNear(analyzer.calculateDirectionalRisk(result.heatmap, {0, 0}), 0.f, "Lone player received opponent risk");
    }

    Matrix<float> heatmap(3, 2, 0.f);
    heatmap[0] = {1.f, 2.f, 3.f};
    heatmap[1] = {4.f, 5.f, 6.f};
    checkNear(analyzer.calculateDirectionalRisk(heatmap, {1, 0}), 9.f, "Directional sum used wrong axes or included diagonals");
}  //----- end of testBoardDimensions


void testInvalidParameters()
// Algorithm : Tests constructor validation by rejecting negative horizons and invalid decay values.
{
    for (float decay : vector<float>{-0.1f, 1.1f, numeric_limits<float>::quiet_NaN()})
    {
        bool rejected = false;

        try
        {
            SpaceRiskAnalyzer analyzer(5, decay);
        }
        catch (const invalid_argument &)
        {
            rejected = true;
        }

        check(rejected, "Invalid decay was accepted");
    }

    bool rejected = false;

    try
    {
        SpaceRiskAnalyzer analyzer(-1);
    }
    catch (const invalid_argument &)
    {
        rejected = true;
    }

    check(rejected, "Negative lookahead was accepted");
}  //----- end of testInvalidParameters


vector<TestCase> getTests()
// Algorithm : Register independent spatial-analysis behavior tests with the component runner.
{
    return {
        {"independent_arrivals", testIndependentArrivals},
        {"contested_territory", testContestedTerritory},
        {"risk_values", testRiskValues},
        {"dead_bodies_and_slots", testDeadBodiesAndSlots},
        {"lookahead_and_static_occupancy", testLookaheadAndStaticOccupancy},
        {"snapshot_independence", testSnapshotIndependence},
        {"board_dimensions", testBoardDimensions},
        {"invalid_parameters", testInvalidParameters}
    };
}  //----- end of getTests
