/*************************************************************************
InputHandler - Reads and writes the competition protocol
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//----- Interface of the module <InputHandler> (file InputHandler.h) -----
#ifndef INPUTHANDLER_H
#define INPUTHANDLER_H

//------------------------------------------------------------------------
// Role of the <InputHandler> module
// Validates input and converts external messages into typed one-based events.
//------------------------------------------------------------------------

//--------------------------------------------------------- System Include
#include <iostream>

//-------------------------------------------------------- Used interfaces
#include "../Structs.h"

//----------------------------------------------------------------- PUBLIC
class InputHandler {
//--------------------------------------------------------- Public Methods
    public:
        static GameConfig readGameConfig(std::istream &input = std::cin);
        // Contract : Read and validate board rules and the already one-based controlled-player ID.

        static std::vector<Coord> readPlayersOrigins(int numPlayers, std::istream &input = std::cin);
        // Contract : Read exactly N coordinate pairs into positions 1..N, leaving index zero unused.

        static GameEvent readEvent(int numPlayers, std::istream &input = std::cin);
        // Contract : Read one complete event; return End on EOF and InputError for malformed messages.

        static void writeMove(Action move, std::ostream &output = std::cout);
        // Contract : Write a valid direction and flush it for the competition runner.
};

#endif //INPUTHANDLER_H
