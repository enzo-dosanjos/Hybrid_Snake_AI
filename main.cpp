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

    printf("> Game Config: W=%d, H=%d, M=%d, N=%d, P=%d\n",
        gameConfig.W, gameConfig.H, gameConfig.M, gameConfig.N, gameConfig.P);
    printf("> Players Origins:\n");
    for (int i = 1; i <= gameConfig.N; ++i) {
        printf(">  Player %d: (%d, %d)\n", i, playersOrigins[i].x, playersOrigins[i].y);
    }

    // Initialize game engine
    GameEngine gameEngine(gameConfig, playersOrigins);

    PlayerSelector playerSelector(gameConfig.P);

    // Initialize observation
    int boardChannels = 9; // 9 Channels for each observation in each cell
    Observation observation(gameConfig, playersOrigins, boardChannels);

    for (int i = 0; i < 8; ++i) {
        Matrix<vector<float>> a = observation.getObservation();
        cout << ">[\n";
        for (size_t i = 0; i < a.size(); ++i) {
            cout << ">  [\n";
            for (size_t j = 0; j < a[i].size(); ++j) {
                cout << ">    [";
                for (size_t k = 0; k < a[i][j].size(); ++k) {
                    cout << a[i][j][k];
                    if (k + 1 < a[i][j].size()) cout << ", ";
                }
                cout << "]";
                if (j + 1 < a[i].size()) cout << ",";
                cout << "\n";
            }
            cout << ">  ]";
            if (i + 1 < a.size()) cout << ",";
            cout << "\n";
        }
        cout << ">]\n";

        int move = 1;
        io.writeMove(move);
        Coord prevHead = gameEngine.getPlayers().at(0).snake.front();
        gameEngine.updateStep(0, move);
        observation.updateBoard(gameEngine.getPlayers(), 0, prevHead, i);

        int oppMove = io.readMove();
        Coord oppPrevHead = gameEngine.getPlayers().at(0).snake.front();
        gameEngine.updateStep(1, oppMove);
        observation.updateBoard(gameEngine.getPlayers(), 1, oppPrevHead, i);
    }
    return 0;
}