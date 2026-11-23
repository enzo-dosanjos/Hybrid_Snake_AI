/*************************************************************************
PlayerSelector - Advances the scheduled player in the shared game state
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//--- Interface of the module <PlayerSelector> (file PlayerSelector.h) ---
#ifndef PLAYERSELECTOR_H
#define PLAYERSELECTOR_H

//------------------------------------------------------------------------
// Role of the <PlayerSelector> module
// Advances scheduled slots, including dead players, without owning another clock.
//------------------------------------------------------------------------

//-------------------------------------------------------- Used interfaces
#include "../Structs.h"

//----------------------------------------------------------------- PUBLIC
class PlayerSelector {
//--------------------------------------------------------- Public Methods
    public:
        void advance(GameState &state) const;
        // Contract : Advance one scheduled slot and increment the round after player N.
};

#endif //PLAYERSELECTOR_H
