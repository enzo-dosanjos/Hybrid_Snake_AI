/*************************************************************************
main - Main function of the program
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

#include <iostream>
#include <algorithm>
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

    for (int i = 0; i < 8; ++i) {
        Matrix<vector<float>> a = observation.getObservation();

        for (size_t x = 0; x < gameConfig.W; ++x) {
            cout << "> [";
            for (size_t y = 0; y < gameConfig.H; ++y) {
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

        int move = 2;
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
        observation.updateBoard(gameEngine.getPlayers(), player, i);
        io.writeMove(move);

        playerSelector.nextPlayer(gameEngine.getPlayers());
        player = playerSelector.getCurrentPlayer();

        pair<string, int> action = io.readAction();
        if (action.first == "death") {
            io.processDeathEvent();
            break;
        }
        int oppMove = io.readMove();
        gameEngine.updateStep(player, oppMove);
        observation.updateBoard(gameEngine.getPlayers(), player, i);
        playerSelector.nextPlayer(gameEngine.getPlayers());

        // Flush output buffer
        cout.flush();
    }
    return 0;
}