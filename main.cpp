/*************************************************************************
main - Main function of the program
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

#include <iostream>
#include <algorithm>
#include <vector>

#include "include/GameEngine.h"
#include "include/Utils.h"
#include "include/Minimax.h"
#include "include/Structs.h"
#include "include/NNAI.h"

using namespace std;


void gameMain() {
    int W, H, M, N, P, X, Y;
    cin >> W >> H;  // size of the board
    cin >> M;       // grow 1 block every M rounds
    cin >> N >> P;  // number of players and player's turn

    vector<Coord> playersOrigins;  // player origins, index 0 is unused
    for (int i = 1; i <= N; i++) {
        cin >> X >> Y;
        playersOrigins[i] = Coord{X, Y};
    }

    GameConfig gameConfig = {W, H, M, N, P};

    // Initialize game engine
    GameEngine gameEngine(gameConfig, playersOrigins);
    Players players;

    // Game loop variables
    string action;
    int playerNb;
    string move;
    string bestMove;
    deque<string> last5Moves;  // todo: move


    int boardChannels = 9; // 9 Channels for each observation in each cell

    // helper class
    Utils helper(gameConfig, boardChannels);

    // NN setup

    // main and target network
    NNAI mainNetwork(gameConfig, boardChannels, 0.1, 0.01, 0.995);
    mainNetwork.loadModelFromFile("model.model");

    // Minimax setup
    Minimax survivor(gameConfig, boardChannels);

    // Merging setup
    float lambda = 0.7;  // change to 0.0 to train miniMax, 1.0 for full DQN and 0.7 for basic use


    while (!gameEngine.terminalState())
    {
        // Read and process moves from all players in the current round
        int playerTurn = gameEngine.playerSelector.currentPlayer;
        if (playerTurn == P)
        {
            vector<float> playerBoardState = gameEngine.observation.getObservation(players);
            vector<float> playerExtraState = helper.computeExtraFeatures(playerBoardState, players, {}, last5Moves, turn);


            // --- Select action using DQN ---
            vector<float> qValues = mainNetwork.forwardPropagation(playerBoardState, playerExtraState);

            // --- Select action using Minimax ---
            unordered_map<string, float> mmScores = survivor.calculateMoveScores(playerBoardState, players, turn);

            // Combine scores with lambda weighting
            map<string, float> combined;
            for (const auto &[move, score] : mmScores) {
                int qIdx = distance(LABEL_ACTIONS.begin(), find_if(LABEL_ACTIONS.begin(), LABEL_ACTIONS.end(),
                    [&move](const pair<int, string> &p) { return p.second == move; }));
                combined[move] = (1 - lambda) * score + lambda * qValues[qIdx];
            }

            // If no valid move, the snake will die so we need to train the network before killing it
            if (combined.empty())
            {
                bestMove = "up";

                // Mark the player as dead as it will die next round
                players[P].playerInfo.alive = false;
                nbPlayerAlive--;
            }
            else
            {
                // Select best combined move
                bestMove = max_element(combined.begin(), combined.end(),
                [](const pair<string, float> &a, const pair<string, float> &b) { return a.second < b.second; })->first;
            }

            last5Moves.push_front(bestMove);

            if (last5Moves.size() > 5)
            {
                last5Moves.pop_back();
            }

            players = gameEngine.snakeUpdate(P, bestMove);

            cout << bestMove << endl;
        }
        else
        {
            // --- Process opponents' moves ---
            cin >> action >> playerNb;
            if (action == "move")
            {
                cin >> move;
                players = gameEngine.snakeUpdate(playerNb+1, move);

            }
            else if (action == "death" && players[playerNb+1].playerInfo.alive)
            {
                cin >> action >> playerNb >> move;

                // Mark the specified player as dead
                players[playerNb].playerInfo.alive = false;
                nbPlayerAlive--;
            }
        }

        // Flush output buffer
        cout.flush();

        // Update player turn
        gameEngine.playerSelector.nextPlayer();
    }
}


int main() {
    gameMain();
    return 0;
}