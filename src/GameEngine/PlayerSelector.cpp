/*************************************************************************
PlayerSelector - Advances player slots and rounds
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

// Implementation of the module <PlayerSelector> (file PlayerSelector.cpp)

//---------------------------------------------------------------- INCLUDE
//------------------------------------------------------- Personal Include
#include "PlayerSelector.h"

//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
void PlayerSelector::advance(GameState &state) const
// Algorithm : Advance one player slot and wrap to the next round after the last player.
{
    state.currentPlayer = (state.currentPlayer % state.config.N) + 1;

    if (state.currentPlayer == 1)
    {
        state.round++;
    }
}  //----- end of advance
