/*************************************************************************
ProofSearch - Resolves bounded wins and losses on the experimental two-player board
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//--- Interface of the module <ProofSearch> (file ProofSearch.h) ---
#ifndef PROOFSEARCH_H
#define PROOFSEARCH_H

//------------------------------------------------------------------------
// Role of the <ProofSearch> module
// Apply the benchmark's 20-turn proof guard to a preferred legal move.
// The search counts the candidate move as the first scheduled player turn.
//------------------------------------------------------------------------

//--------------------------------------------------------- System Include
#include <array>
#include <map>

//------------------------------------------------------- Personal Include
#include "../../src/GameEngine/GameEngine.h"

//----------------------------------------------------------------- PUBLIC
class ProofSearch {
    public:
        static Action select(const GameEngine &engine, const std::map<Action, float> &scores, Action preferred);
        // Contract : Scores contain all legal moves from the scheduled player's perspective.

    private:
        explicit ProofSearch(const GameState &state);
        bool canForce(int player, int target, int round, int remaining);

        static constexpr int CAPACITY = 256;
        std::array<bool, CAPACITY> occupied{};
        std::array<std::array<int, CAPACITY>, 2> body{};
        std::array<int, 2> head{};
        std::array<int, 2> tail{};
        std::array<int, 4> offsets{};
        int growthInterval;
};

#endif // PROOFSEARCH_H
