/*************************************************************************
MinimaxParamsFile - Reads and writes a single set of Minimax parameters
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//-- Implementation of the module <MinimaxParamsFile> (file MinimaxParamsFile.cpp) --

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <stdexcept>
#include <unistd.h>

using namespace std;

//------------------------------------------------------- Personal Include
#include "MinimaxParamsFile.h"

//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
MinimaxParams loadMinimaxParams(const string &filename)
// Algorithm : Parse one file and use the Minimax constructor to enforce its parameter contract.
{
    ifstream input(filename);
    if (!input)
    {
        throw runtime_error("Cannot open Minimax parameters " + filename);
    }

    input.imbue(locale::classic());
    MinimaxParams params;
    Rewards &weights = params.rewards;
    if (!(input >> weights.selfSpace >> weights.oppSpace >> weights.selfBlockRisk >> weights.oppBlockRisk
                >> weights.selfHeatRisk >> weights.oppHeatRisk >> weights.snakeLength >> params.depth) ||
        (input >> ws).peek() != char_traits<char>::eof() || input.bad() || !input.eof())
    {
        throw invalid_argument("Malformed Minimax parameters in " + filename);
    }

    Minimax{params};
    return params;
}  //----- end of loadMinimaxParams


void saveMinimaxParams(const MinimaxParams &params, const string &filename)
// Algorithm : Write round-trip precision to a temporary sibling before replacing the destination.
{
    Minimax{params};
    string temporary = filename + ".tmp.XXXXXX";
    int descriptor = mkstemp(temporary.data());
    if (descriptor == -1)
    {
        throw runtime_error("Cannot create temporary Minimax parameter file for " + filename);
    }

    try
    {
        if (close(descriptor) != 0)
        {
            throw runtime_error("Cannot close temporary Minimax parameter file for " + filename);
        }

        ofstream output(temporary);
        output.imbue(locale::classic());
        const Rewards &weights = params.rewards;
        output << setprecision(numeric_limits<double>::max_digits10)
               << weights.selfSpace << ' ' << weights.oppSpace << ' '
               << weights.selfBlockRisk << ' ' << weights.oppBlockRisk << ' '
               << weights.selfHeatRisk << ' ' << weights.oppHeatRisk << ' '
               << weights.snakeLength << '\n' << params.depth << '\n';
        output.close();
        if (!output)
        {
            throw runtime_error("Cannot write Minimax parameters to " + filename);
        }

        filesystem::rename(temporary, filename);
    }
    catch (...)
    {
        error_code ignored;
        filesystem::remove(temporary, ignored);
        throw;
    }
}  //----- end of saveMinimaxParams
