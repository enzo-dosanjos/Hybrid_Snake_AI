/*************************************************************************
Utils - Utility functions
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//------- Interface of the module <Utils> (file Utils.h) -------
#ifndef UTILS_H
#define UTILS_H

//------------------------------------------------------------------------
// Role of the <Utils> module
// Provides utility functions for the game engine
//------------------------------------------------------------------------

//-------------------------------------------------------- Used interfaces
#include "Structs.h"

//-------------------------------------------------------------- Constants

//------------------------------------------------------------------ Types

//----------------------------------------------------------------- PUBLIC
static Coord computeNewCoord(const Coord &currentHead, const int &move)
// Usage :
//
// Contract :
// Computes the new coordinates of the snake's head based on the current head and the move
{
    Coord newHead = {
        currentHead.x + DIRECTIONS[move].x,
        currentHead.y + DIRECTIONS[move].y
    };

    printf("> Moving from (%d, %d) to (%d, %d) using move %d (%s)\n",
        currentHead.x, currentHead.y,
        newHead.x, newHead.y,
        move,
        ACTIONS[move].c_str()
    );

    return newHead;
}  // ----- End of computeNewCoord

static int computeOppositeDir(const int &move)
// Usage :
//
// Contract :
// Computes the opposite direction of the given move
{
    return (move + 2) % 4;
}  // ----- End of computeOppositeDir

inline bool isInbound(Coord coord, const GameConfig &config)
// Usage :
//
// Contract :
// Check if the coordinates are in the board's bounds
{
    return coord.x >= 0 && coord.x < config.W &&
        coord.y >= 0 && coord.y < config.H;
}  //----- End of isInbound


#endif //UTILS_H
