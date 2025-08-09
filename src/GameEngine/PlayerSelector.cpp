/*************************************************************************
PlayerSelector - Handler for player turn selection
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

// Implementation of the module <PlayerSelector> (file PlayerSelector.cpp)

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <iostream>

using namespace std;

//------------------------------------------------------- Personal Include
#include "PlayerSelector.h"

//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
int PlayerSelector::nextPlayer(int numPlayers, const Players &playerState)
// Algorithm : Update the current player to the next alive player
{
    do {
        currentPlayer = (currentPlayer % numPlayers) + 1;
    } while (!playerState.at(currentPlayer).playerInfo.alive);

    return currentPlayer;
}  // ----- End of nextPlayer