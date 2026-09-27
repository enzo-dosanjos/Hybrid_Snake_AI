/*************************************************************************
GameTest - Provides state comparison and reference occupancy checks
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//--------- Interface of the module <GameTest> (file GameTest.h) ---------

#ifndef GAMETEST_H
#define GAMETEST_H

//------------------------------------------------------------------------
// Role of the <GameTest> module
// Provides state comparison and reference occupancy checks
//------------------------------------------------------------------------

//-------------------------------------------------------- Used interfaces
#include "Test.h"
#include "src/GameEngine/GameEngine.h"


//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
inline void checkState(const GameState &actual, const GameState &expected)
// Algorithm : Checks snapshot equality across configuration, scheduling, history, snakes and player metadata.
{
    check(actual.config.W == expected.config.W && actual.config.H == expected.config.H &&
          actual.config.M == expected.config.M && actual.config.N == expected.config.N &&
          actual.config.P == expected.config.P, "Configuration changed");

    check(actual.round == expected.round && actual.currentPlayer == expected.currentPlayer &&
          actual.lastMoves == expected.lastMoves, "Clock or history changed");
    check(actual.players.size() == expected.players.size(), "Player count changed");

    for (const auto &[id, player] : expected.players)
    {
        const auto &other = actual.players.at(id);

        check(other.snake == player.snake, "Snake changed for player " + std::to_string(id));
        check(other.playerInfo.alive == player.playerInfo.alive &&
              other.playerInfo.origin == player.playerInfo.origin &&
              other.playerInfo.actionMask == player.playerInfo.actionMask,
              "Player metadata changed");
    }
}  //----- end of checkState


inline bool occupied(const GameState &state, const Coord &cell)
// Algorithm : Determines reference occupancy by scanning every stored snake segment, including retained dead bodies.
{
    for (const auto &[id, player] : state.players)
    {
        for (const Coord &part : player.snake)
        {
            if (part == cell)
            {
                return true;
            }
        }
    }

    return false;
}  //----- end of occupied

#endif //GAMETEST_H
