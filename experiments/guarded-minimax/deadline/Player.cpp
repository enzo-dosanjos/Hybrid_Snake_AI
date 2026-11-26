/*************************************************************************
Player - Runs the two-player deadline benchmark over the competition protocol
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//--------- Implementation of the module <Player> (file Player.cpp) ---------

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <algorithm>
#include <iostream>
#include <string_view>

using namespace std;

//------------------------------------------------------- Personal Include
#include "../../tactical-guard/ProofSearch.h"
#include "../../../src/GameEngine/InputHandler.h"
#include "../../../src/Minimax/Minimax.h"
#include "../../../src/Minimax/MinimaxParamsFile.h"

//----------------------------------------------------------------- PUBLIC
int main(int argc, char **argv)
// Algorithm : Reply on the configured seat's turn using either guarded or ordinary tuned Minimax.
{
    try
    {
        if (argc != 2 || (string_view(argv[1]) != "guard" && string_view(argv[1]) != "opponent"))
        {
            throw invalid_argument("Usage: Player guard|opponent");
        }

        bool guarded = string_view(argv[1]) == "guard";
        Minimax search(loadMinimaxParams("experiments/minimax-cpu-pilot/winner.txt"));
        auto config = InputHandler::readGameConfig();
        GameEngine engine(config, InputHandler::readPlayersOrigins(config.N));

        while (!engine.isTerminalFor(config.P))
        {
            if (engine.getState().currentPlayer != config.P)
            {
                auto event = InputHandler::readEvent(config.N);
                if (event.type == GameEventType::End || event.type == GameEventType::Death)
                {
                    break;
                }
                if (event.type != GameEventType::Move || event.player != engine.getState().currentPlayer)
                {
                    throw invalid_argument("Expected the scheduled opponent's move");
                }

                engine.updateStep(*event.action);
                continue;
            }

            auto scores = search.calculateMoveScores(engine);
            if (scores.empty())
            {
                break;
            }

            auto best = max_element(scores.begin(), scores.end(), [](const auto &left, const auto &right)
            {
                return left.second < right.second;
            });

            Action move = guarded ? ProofSearch::select(engine, scores, best->first) : best->first;
            InputHandler::writeMove(move);
            engine.updateStep(move);
        }
    }
    catch (const exception &error)
    {
        cerr << error.what() << endl;
        return 1;
    }

    return 0;
}  //----- end of main
