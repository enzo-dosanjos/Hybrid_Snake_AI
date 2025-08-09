/*************************************************************************
InputHandler - todo
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//----- Interface of the module <InputHandler> (file InputHandler.h) -----
#ifndef INPUTHANDLER_H
#define INPUTHANDLER_H

//------------------------------------------------------------------------
// Role of the <InputHandler> module
// todo
//------------------------------------------------------------------------

//-------------------------------------------------------- Used interfaces
#include "../Structs.h"

//-------------------------------------------------------------- Constants

//------------------------------------------------------------------ Types

//----------------------------------------------------------------- PUBLIC
class InputHandler {
//--------------------------------------------------------- Public Methods
    public:
        static GameConfig readGameConfig();

        static std::vector<Coord> readPlayersOrigins(int numPlayers);

        static std::pair<std::string, int> readAction();

        static int readMove();

        void processDeathEvent();

        void writeMove(const int move);
};



#endif //INPUTHANDLER_H
