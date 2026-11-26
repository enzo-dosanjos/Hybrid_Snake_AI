/*************************************************************************
ProofSearch - Resolves bounded wins and losses on the experimental two-player board
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//--- Implementation of the module <ProofSearch> (file ProofSearch.cpp) ---

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

using namespace std;

//------------------------------------------------------- Personal Include
#include "ProofSearch.h"

//----------------------------------------------------------------- PUBLIC
Action ProofSearch::select(const GameEngine &engine, const map<Action, float> &scores, Action preferred)
// Algorithm : Prefer proven wins, then unresolved moves, keeping the preferred move when eligible.
{
    if (scores.empty() || !scores.count(preferred))
    {
        throw invalid_argument("The proof guard requires a legal preferred move");
    }

    const float terminal = numeric_limits<float>::max() / 4.f;
    auto best = max_element(scores.begin(), scores.end(), [](const auto &left, const auto &right)
    {
        return left.second < right.second;
    });

    if (scores.size() == 1 || abs(best->second) == terminal)
    {
        return best->first;
    }

    int rootPlayer = engine.getState().currentPlayer - 1;
    ActionMask wins;
    ActionMask unresolved;

    for (const auto &[move, value] : scores)
    {
        if (value == -terminal)
        {
            continue;
        }

        auto child = engine.successor(move);
        const auto &state = child.getState();
        ProofSearch board(state);
        int nextPlayer = state.currentPlayer - 1;
        int index = static_cast<int>(move);

        if (board.canForce(nextPlayer, rootPlayer, state.round, 19))
        {
            wins.set(index);
        }
        else if (!board.canForce(nextPlayer, 1 - rootPlayer, state.round, 19))
        {
            unresolved.set(index);
        }
    }

    const auto &eligible = wins.any() ? wins : unresolved;
    if (eligible.none() || eligible[static_cast<int>(preferred)])
    {
        return preferred;
    }

    // Map order gives the same first-action tie break as MinimaxAgent.
    float bestScore = -numeric_limits<float>::infinity();
    for (const auto &[move, value] : scores)
    {
        if (eligible[static_cast<int>(move)] && value > bestScore)
        {
            bestScore = value;
            preferred = move;
        }
    }

    return preferred;
}  //----- end of select


//---------------------------------------------------------------- PRIVATE
ProofSearch::ProofSearch(const GameState &state)
    : growthInterval(state.config.M)
// Algorithm : Surround the board with walls and store each living snake from tail to head.
{
    int stride = state.config.W + 2;
    if (state.config.N != 2 || stride * (state.config.H + 2) > CAPACITY)
    {
        throw invalid_argument("Unsupported proof board dimensions or player count");
    }

    offsets = {-1, -stride, 1, stride};
    occupied.fill(true);
    for (int y = 1; y <= state.config.H; y++)
    {
        fill_n(occupied.begin() + y*stride + 1, state.config.W, false);
    }

    for (int player = 0; player < 2; player++)
    {
        const auto &snake = state.players.at(player + 1).snake;
        for (auto part = snake.rbegin(); part != snake.rend(); ++part)
        {
            int cell = (part->y + 1)*stride + part->x + 1;
            occupied[cell] = true;
            body[player][head[player]++] = cell;
        }
        head[player]--;
    }
}  //----- end of ProofSearch


bool ProofSearch::canForce(int player, int target, int round, int remaining)
// Algorithm : Require one winning move on the target's turn and a win after every opponent reply.
{
    if (remaining == 0)
    {
        return false;
    }

    bool ownTurn = player == target;
    bool release = round % growthInterval != 0;
    int currentHead = body[player][head[player]];
    int currentTail = body[player][tail[player]];

    for (int offset : offsets)
    {
        int next = currentHead + offset;
        if (occupied[next])
        {
            continue;
        }

        // Collision is checked before the tail moves, even on non-growth rounds.
        occupied[next] = true;
        body[player][++head[player]] = next;
        if (release)
        {
            occupied[currentTail] = false;
            tail[player]++;
        }

        // Player index one closes the round in this two-player representation.
        bool wins = canForce(1 - player, target, round + player, remaining - 1);

        if (release)
        {
            tail[player]--;
            occupied[currentTail] = true;
        }
        head[player]--;
        occupied[next] = false;

        if (wins == ownTurn)
        {
            return ownTurn;
        }
    }

    return !ownTurn;
}  //----- end of canForce
