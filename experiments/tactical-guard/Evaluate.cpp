/*************************************************************************
Evaluate - Plays the bounded proof guard against tuned Minimax on one layout
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//--------- Implementation of the module <Evaluate> (file Evaluate.cpp) ---------

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <algorithm>
#include <iostream>
#include <memory>

using namespace std;

//------------------------------------------------------- Personal Include
#include "ProofSearch.h"
#include "../../src/ApplicationModel.h"
#include "../../src/DQN/ModelPersistence.h"
#include "../../src/GameEngine/InputHandler.h"
#include "../../src/Minimax/MinimaxParamsFile.h"

//----------------------------------------------------------------- PUBLIC
int main(int argc, char **argv)
// Algorithm : Replay one protocol setup until a player is trapped and print the guarded seat's result.
{
    try
    {
        if (argc > 2)
        {
            throw invalid_argument("Usage: Evaluate [ranked.model] < setup");
        }

        auto config = InputHandler::readGameConfig();
        GameEngine engine(config, InputHandler::readPlayersOrigins(config.N));
        auto params = loadMinimaxParams("experiments/minimax-cpu-pilot/winner.txt");
        Minimax search(params);
        unique_ptr<DQN> learner;
        unique_ptr<HybridAgent> hybrid;

        if (argc == 2)
        {
            learner = make_unique<DQN>(makeApplicationModelConfig(config.N), mt19937{});
            ModelPersistence{}.loadDQN(*learner, argv[1]);
            hybrid = make_unique<HybridAgent>(*learner, params, 1.f, false, 1.f);
        }

        for (int turns = 1; ; turns++)
        {
            int player = engine.getState().currentPlayer;
            auto scores = search.calculateMoveScores(engine);
            if (scores.empty())
            {
                cout << (player != config.P) << " " << turns << endl;
                break;
            }

            auto best = max_element(scores.begin(), scores.end(), [](const auto &left, const auto &right)
            {
                return left.second < right.second;
            });
            Action move = best->first;

            if (player == config.P)
            {
                Action preferred = hybrid ? hybrid->selectAction(engine).value() : move;
                move = ProofSearch::select(engine, scores, preferred);
            }

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
