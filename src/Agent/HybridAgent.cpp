/*************************************************************************
HybridAgent - todo
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//-- Implementation of the module <HybridAgent> (file HybridAgent.cpp) --

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <algorithm>
#include <iostream>

using namespace std;

//------------------------------------------------------- Personal Include
#include "../include/HybridAgent.h"

//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
string HybridAgent::selectAction(
    const vector<float> &boardState,
    const vector<float> &extraFeatures,
    const Players &playerState,
    const deque<string> &last5Moves,
    int turn
) {
    // --- Select action using DQN ---
    vector<float> qValues = dqn.forwardPropagation(boardState, extraFeatures);

    // --- Select action using Minimax ---
    unordered_map<string, float> mmScores = minimax.calculateMoveScores(boardState);

    // Combine scores with lambda weighting
    map<string, float> combined;
    for (const auto &[move, score] : mmScores) {
        int qIdx = distance(ACTIONS.begin(), find_if(ACTIONS.begin(), ACTIONS.end(),
            [&move](const pair<int, string> &p) { return p.second == move; }));
        combined[move] = (1 - lambda) * score + lambda * qValues[qIdx];
    }

    return max_element(combined.begin(), combined.end(),
        [](const pair<string, float> &a, const pair<string, float> &b) {
            return a.second < b.second;
        })->first;
}  // ----- End of selectAction