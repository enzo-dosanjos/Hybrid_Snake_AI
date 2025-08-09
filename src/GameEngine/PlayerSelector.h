/*************************************************************************
PlayerSelector - Handler for player turn selection
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//--- Interface of the module <PlayerSelector> (file PlayerSelector.h) ---
#ifndef PLAYERSELECTOR_H
#define PLAYERSELECTOR_H

//------------------------------------------------------------------------
// Role of the <PlayerSelector> module
// The PlayerSelector module handles the selection of player turns
//------------------------------------------------------------------------

//-------------------------------------------------------- Used interfaces

//-------------------------------------------------------------- Constants

//------------------------------------------------------------------ Types
#include "../Structs.h"

//----------------------------------------------------------------- PUBLIC
class PlayerSelector
{
//--------------------------------------------------------- Public Methods
    public:
        int nextPlayer(
            const Players &playerState
        );
        // Usage :
        //
        // Contract :
        //

        bool myTurn() const
        {
            return currentPlayer == playerTurn;
        }
        // Usage :
        //
        // Contract :
        //

//--------------------------------------------------------- SETTERS/GETTERS
        int getCurrentPlayer() const
        {
            return currentPlayer;
        }

//---------------------------------------------- Constructors - destructor
        PlayerSelector(const int p_numPlayers, const int p_playerTurn
        ) : numPlayers(p_numPlayers), currentPlayer(1), playerTurn(p_playerTurn)
        {
            #ifdef MAP
                std::cout << "Calling the constructor of PlayerSelector" << std::endl;
            #endif
        }

//-------------------------------------------------------------- PROTECTED

//---------------------------------------------------------------- PRIVATE
    private:
        int numPlayers;
        int currentPlayer;
        int playerTurn;
};



#endif //PLAYERSELECTOR_H
