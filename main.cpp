/*************************************************************************
main - Main function of the program
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

#include <iostream>
#include <algorithm>
#include <random>
#include <vector>

#include "src/GameEngine/GameEngine.h"
#include "src/Structs.h"
#include "src/GameEngine/InputHandler.h"
#include "src/GameEngine/Observation.h"
#include "src/GameEngine/PlayerSelector.h"

using namespace std;


int main() {
    InputHandler io;

    // Read game configuration
    GameConfig gameConfig = io.readGameConfig();
    vector<Coord> playersOrigins = io.readPlayersOrigins(gameConfig.N);

    // Initialize game engine
    GameEngine gameEngine(gameConfig, playersOrigins);

    PlayerSelector playerSelector(gameConfig.N, gameConfig.P);

    // Initialize observation
    int boardChannels = 9; // 9 Channels for each observation in each cell
    Observation observation(gameConfig, playersOrigins, boardChannels);

    Matrix<vector<float>> a = observation.getObservation();

    for (int x = 0; x < gameConfig.W; ++x) {
        cout << "> [";
        for (int y = 0; y < gameConfig.H; ++y) {
            cout << "[";
            const auto& channels = a[y][x];
            for (size_t k = 0; k < channels.size(); ++k) {
                cout << channels[k];
                if (k + 1 < channels.size()) cout << ", ";
            }
            cout << "]";
            if (y + 1 < gameConfig.H) cout << ", ";
        }
        cout << "]\n";
    }

    while (!gameEngine.isTerminalState()) {
        int move;
        if (playerSelector.getCurrentPlayer() == gameConfig.P)
        {
            // Randomly choose a move from the action mask
            vector<int> validMoves;
            for (size_t i = 0; i < ACTIONS.size(); i++)
            {
                cout << "> Action mask for action " << ACTIONS[i] << ": " << gameEngine.getActionMask(gameConfig.P)[i] << "\n";
                if (gameEngine.getActionMask(gameConfig.P)[i])
                {
                    validMoves.push_back(i);
                }
            }

            if (validMoves.empty()) {
                // No valid moves: treat as death and break
                io.processDeathEvent();
                break;
            }

            random_device rd;
            mt19937 gen(rd());
            uniform_int_distribution<> dis(0, validMoves.size() - 1);
            move = validMoves[dis(gen)];

            io.writeMove(move);
        }
        else
        {
            pair<string, int> action = io.readAction();
            if (action.first == "death") {
                gameEngine.playerDied(action.second+1); // player numbers are 1-indexed
                io.processDeathEvent();
                continue;
            }
            move = io.readMove();
        }

        int player = playerSelector.getCurrentPlayer();
        gameEngine.updateStep(player, move);
        // display players
        for (int p = 1; p <= gameConfig.N; ++p) {
            const auto &snake = gameEngine.getPlayers().at(p).snake;
            cout << "> Player " << p << " snake: ";
            for (const auto &part : snake) {
                cout << "(" << part.x << ", " << part.y << ") ";
            }
            cout << "\n";
        }
        int round = gameEngine.getRound();
        observation.updateBoard(gameEngine.getPlayers(), player, round);

        Matrix<vector<float>> a = observation.getObservation();
        for (int x = 0; x < gameConfig.W; ++x) {
            cout << "> [";
            for (int y = 0; y < gameConfig.H; ++y) {
                cout << "[";
                const auto& channels = a[y][x];
                for (size_t k = 0; k < channels.size(); ++k) {
                    cout << channels[k];
                    if (k + 1 < channels.size()) cout << ", ";
                }
                cout << "]";
                if (y + 1 < gameConfig.H) cout << ", ";
            }
            cout << "]\n";
        }

        playerSelector.nextPlayer(gameEngine.getPlayers());

        // Flush output buffer
        cout.flush();
    }

    return 0;
}