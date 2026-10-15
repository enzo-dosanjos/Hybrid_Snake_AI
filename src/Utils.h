/*************************************************************************
Utils - Utility functions for game rules and coordinates
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//------- Interface of the module <Utils> (file Utils.h) -------
#ifndef UTILS_H
#define UTILS_H

//------------------------------------------------------------------------
// Role of the <Utils> module
// Provides shared configuration checks and coordinate operations.
//------------------------------------------------------------------------

//--------------------------------------------------------- System Include
#include <stdexcept>

//-------------------------------------------------------- Used interfaces
#include "Structs.h"

//-------------------------------------------------------------- Constants

//------------------------------------------------------------------ Types

//----------------------------------------------------------------- PUBLIC
inline void validateGameConfig(const GameConfig &config)
// Algorithm : Reject invalid dimensions, growth periods and player identities before initializing a game.
{
    if (config.W <= 0 || config.H <= 0 || config.M <= 0 || config.N <= 0 ||
        config.P < 1 || config.P > config.N ||
        config.N > static_cast<long long>(config.W) * config.H)
    {
        throw std::invalid_argument("Invalid game configuration");
    }
}  //----- end of validateGameConfig


inline Coord computeNewCoord(const Coord &currentHead, const int &move)
// Algorithm : Add the selected direction to the current head, rejecting an invalid action index.
{
    const Coord &direction = DIRECTIONS.at(move);
    Coord newHead = {
        currentHead.x + direction.x,
        currentHead.y + direction.y
    };

    return newHead;
}  //----- end of computeNewCoord


inline Coord computeNewCoord(const Coord &currentHead, Action move)
// Algorithm : Use the shared direction table for a typed action.
{
    return computeNewCoord(currentHead, static_cast<int>(move));
}  //----- end of computeNewCoord


inline int computeOppositeDir(const int &move)
// Algorithm : Validate the action index and rotate it by two directions.
{
    DIRECTIONS.at(move);
    return (move + 2) % 4;
}  //----- end of computeOppositeDir


inline Action computeOppositeDir(Action move)
// Algorithm : Convert the opposite direction index back into a typed action.
{
    return static_cast<Action>(computeOppositeDir(static_cast<int>(move)));
}  //----- end of computeOppositeDir


inline bool isInbound(Coord coord, const GameConfig &config)
// Algorithm : Check if the coordinates are in the board's bounds
{
    return coord.x >= 0 && coord.x < config.W && coord.y >= 0 && coord.y < config.H;
}  //----- end of isInbound

#endif //UTILS_H
