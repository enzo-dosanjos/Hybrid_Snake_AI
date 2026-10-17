/*************************************************************************
Observation - Encodes a game snapshot as board channels
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//------ Interface of the module <Observation> (file Observation.h) ------
#ifndef OBSERVATION_H
#define OBSERVATION_H


//------------------------------------------------------------------------
// Role of the <Observation> module
// This module is responsible for managing the board representation used
// by the model's CNN. It provides methods to build an independent
// channels-first board from an explicit perspective
//------------------------------------------------------------------------

//-------------------------------------------------------- Used interfaces
#include "../Structs.h"

//----------------------------------------------------------------- PUBLIC
class Observation {
//--------------------------------------------------------- Public Methods
    public:
        BoardTensor encode(const GameState &state, int perspective) const;
        // Contract : Encode [channel][y][x]; use four channels per player plus free cells; own slot first, then ascending opponent IDs.
};

#endif //OBSERVATION_H
