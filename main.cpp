/*************************************************************************
main - Plays the competition game
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//--------- Implementation of the module <main> (file main.cpp) ---------

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

using namespace std;

//------------------------------------------------------- Personal Include
#include "src/GameEngine/GameEngine.h"
#include "src/GameEngine/InputHandler.h"

//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
int main()
// Algorithm : Choose legal moves on our turns and apply external events once in scheduled-player order.
{
    try
    {
        GameConfig config = InputHandler::readGameConfig();
        auto origins = InputHandler::readPlayersOrigins(config.N);
        GameEngine engine(config, origins);
        mt19937 gen(random_device{}());
        int lastDeath = 0;

        while (!engine.isTerminalFor(config.P))
        {
            if (engine.getState().currentPlayer == config.P)
            {
                vector<Action> moves;
                ActionMask mask = engine.getActionMask(config.P);

                for (size_t action = 0; action < ACTIONS.size(); action++)
                {
                    if (mask[action])
                    {
                        moves.push_back(static_cast<Action>(action));
                    }
                }

                if (moves.empty())
                {
                    engine.playerDied(config.P);
                    break;
                }

                uniform_int_distribution<size_t> choice(0, moves.size() - 1);
                Action move = moves[choice(gen)];
                InputHandler::writeMove(move);
                engine.updateStep(move);
                continue;
            }

            GameEvent event = InputHandler::readEvent(config.N);

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
    }
    catch (const exception &error)
    {
        cerr << error.what() << endl;
        return 1;
    }

    return 0;
}  //----- end of main
