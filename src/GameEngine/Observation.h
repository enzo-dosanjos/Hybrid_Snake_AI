/*************************************************************************
Observation - Encodes a game snapshot as board channels
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
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
        // Contract : Encode [channel][y][x]; retain dead bodies and mark length-one snakes as head and tail.
};

#endif //OBSERVATION_H
