/*************************************************************************
main - Plays the competition game
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//--------- Implementation of the module <main> (file main.cpp) ---------

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <iostream>
#include <optional>
#include <random>
#include <stdexcept>
#include <string_view>

using namespace std;

//------------------------------------------------------- Personal Include
#include "src/Agent/MinimaxAgent.h"
#include "src/ApplicationModel.h"
#include "src/DQN/ModelPersistence.h"
#include "src/GameEngine/GameEngine.h"
#include "src/GameEngine/InputHandler.h"
#include "src/Minimax/MinimaxParamsFile.h"

//---------------------------------------------------------------- PRIVATE
static void playGame(GameEngine &engine, Agent &agent)
// Algorithm : Use the selected agent and apply external events once in scheduled-player order.
{
    int player = engine.getP();
    int lastDeath = 0;

    while (!engine.isTerminalFor(player))
    {
        if (engine.getState().currentPlayer == player)
        {
            optional<Action> move = agent.selectAction(engine);

            if (!move)
            {
                engine.playerDied(player);
                engine.finishTurn();
                break;
            }

            InputHandler::writeMove(*move);
            engine.updateStep(*move);
            continue;
        }

        GameEvent event = InputHandler::readEvent(engine.getN());

        if (event.type == GameEventType::End)
        {
            break;
        }

        if (event.type == GameEventType::InputError)
        {
            throw invalid_argument(event.error);
        }

        // A collision produces both death and move messages in snake.py; its turn was already counted.
        if (event.type == GameEventType::Move && event.player == lastDeath)
        {
            lastDeath = 0;
            continue;
        }

        lastDeath = 0;

        if (event.player != engine.getState().currentPlayer)
        {
            throw invalid_argument("Game event does not match the scheduled player");
        }

        if (event.type == GameEventType::Death)
        {
            engine.playerDied(event.player);
            engine.finishTurn();
            lastDeath = event.player;
        }
        else
        {
            engine.updateStep(*event.action);
        }
    }
}  //----- end of playGame


//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
int main(int argc, char **argv)
// Algorithm : Select a play policy, load requested parameters and run the competition protocol.
{
    try
    {
        string_view mode = argc == 1 ? "minimax" : argv[1];
        const char *paramsFile = nullptr;
        if (((mode == "minimax" && argc == 4) || (mode == "hybrid" && argc == 6)) &&
            string_view(argv[argc - 2]) == "--minimax-params")
        {
            paramsFile = argv[argc - 1];
            argc -= 2;
        }

        bool minimax = mode == "minimax" && argc <= 2;
        bool dqn = mode == "dqn" && argc == 3;
        bool hybrid = mode == "hybrid" && argc == 4;
        if (!minimax && !dqn && !hybrid)
        {
            throw invalid_argument("Usage: AI [minimax [--minimax-params FILE] | dqn CHECKPOINT | "
                                   "hybrid CHECKPOINT LAMBDA [--minimax-params FILE]]");
        }

        optional<float> hybridWeight;
        if (hybrid)
        {
            hybridWeight = parseHybridWeight(argv[3]);
        }

        MinimaxParams params = paramsFile ? loadMinimaxParams(paramsFile) : MinimaxParams{};
        GameConfig config = InputHandler::readGameConfig();
        auto origins = InputHandler::readPlayersOrigins(config.N);
        GameEngine engine(config, origins);

        if (minimax)
        {
            MinimaxAgent agent(params);
            playGame(engine, agent);
        }
        else
        {
            DQN learner(makeApplicationModelConfig(config.N), mt19937{});
            ModelPersistence{}.loadDQN(learner, argv[2]);
            auto agent = makeNeuralAgent(learner, hybridWeight, false, params);
            playGame(engine, *agent);
        }
    }
    catch (const exception &error)
    {
        cerr << error.what() << endl;
        return 1;
    }

    return 0;
}  //----- end of main
