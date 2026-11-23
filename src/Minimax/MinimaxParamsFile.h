/*************************************************************************
MinimaxParamsFile - Reads and writes a single set of Minimax parameters
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//-- Interface of the module <MinimaxParamsFile> (file MinimaxParamsFile.h) --
#ifndef MINIMAXPARAMSFILE_H
#define MINIMAXPARAMSFILE_H

//------------------------------------------------------------------------
// Role of the <MinimaxParamsFile> module
// Share the optimizer's parameter format with play and training.
//------------------------------------------------------------------------

//--------------------------------------------------------- System Include
#include <string>

//-------------------------------------------------------- Used interfaces
#include "Minimax.h"

//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
MinimaxParams loadMinimaxParams(const std::string &filename);
// Contract : Read exactly seven finite weights and a positive depth.

void saveMinimaxParams(const MinimaxParams &params, const std::string &filename);
// Contract : Replace the destination with a complete parameter file.

#endif //MINIMAXPARAMSFILE_H
