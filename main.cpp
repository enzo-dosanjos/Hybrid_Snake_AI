/*************************************************************************
main - Main function of the program
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

#include <iostream>
#include <algorithm>

// TCP communication
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/select.h>
#include <thread>    // Pour std::this_thread
#include <cstring>

using namespace std;

#include "include/Utils.h"
#include "include/Minimax.h"
#include "include/Structs.h"
#include "include/NNAI.h"


void send_transition(const Transition &t)
// Algorithm : Send transition to the TCP training server
{
    const int max_retries = 3;
    int retry_count = 0;

    while(retry_count < max_retries) {
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if(sock < 0) {
            cout << "> Socket error (retry " << retry_count << "): " << strerror(errno) << endl;
            continue;
        }

        // timeout configuration
        timeval timeout{1, 0}; // 1 seconde
        setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));

        sockaddr_in serv_addr{};
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(12345);
        inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr);

        if(connect(sock, (sockaddr*)&serv_addr, sizeof(serv_addr))) {
            cout << "> Connect error (retry " << retry_count << "): " << strerror(errno) << endl;
            close(sock);
            retry_count++;
            this_thread::sleep_for(10ms);
            continue;
        }

        string serialized;
        try {
            serialized = t.serialize();
        } catch(const exception& e) {
            cout << "> Serialization error: " << e.what() << endl;
            close(sock);
            return;
        }

        uint32_t msg_size = htonl(serialized.size());

        // send transition size
        ssize_t sent = send(sock, &msg_size, sizeof(msg_size), 0);
        if(sent != sizeof(msg_size)) {
            cout << "> Header send error: " << strerror(errno) << endl;
            close(sock);
            retry_count++;
            continue;
        }

        // send transition
        const char* data = serialized.data();
        size_t total_sent = 0;
        while(total_sent < serialized.size()) {
            const ssize_t n = send(sock, data + total_sent, serialized.size() - total_sent, 0);
            if(n <= 0) {
                cout << "> Data send error: " << strerror(errno) << endl;
                break;
            }
            total_sent += n;
        }

        if(total_sent == serialized.size()) {
            cout << "> Transition successfully sent (" << serialized.size() << " bytes)" << endl;
            close(sock);
            return;
        }

        close(sock);
        retry_count++;
    }

    cout << "> Failed to send transition after " << max_retries << " attempts" << endl;
}  // ----- End of send_transition


PlayersData snakeUpdate(const PlayersData &Players, const int &currentPlayer,
                        const int &M, const int &round, const string &move)
// Algorithm : Update the snake's position based on the chosen action
{

    PlayersData updatedPlayers = Players;

    // --- Update snake based on chosen action ---
    Coord currentHead = updatedPlayers[currentPlayer].first.front();
    updatedPlayers[currentPlayer].first.emplace_front(currentHead.first + DIRECTIONS.at(move).first,
                                                      currentHead.second + DIRECTIONS.at(move).second
                                                     );

    // --- Remove tail segments if it's not a growth round ---
    if (round % M != 0)
    {
        updatedPlayers[currentPlayer].first.pop_back();
    }

    return updatedPlayers;
}  // ----- End of snakeUpdate


void optimiseMain() {
    int W, H, M, N, P, X, Y;
    cin >> W >> H;  // size of the board
    cin >> M;       // grow 1 block every M rounds
    cin >> N >> P;  // number of players and player's turn


    // Store player nb (int), parts of their body (deque<coord> and data like
    // the origin of the snake, the accessible space and their risk of being blocked
    PlayersData Players;
    for (int i = 1; i <= N; i++) {
        cin >> X >> Y;
        Players[i].first.emplace_front(X, Y);  // push the head of the player

        // Initialise Player's data
        Players[i].second.origin = make_pair(X, Y);  // origin of the player
        Players[i].second.alive = true;
    }


    // Game loop variables
    string action;
    int playerNb;
    string move;
    int round = 0;
    int turn = 0;
    int nbPlayerAlive = N;
    string bestMove;


    int boardChannels = 6; // 6 Channels for each case of the board

    // helper class
    Utils helper(W, H, M, N, P, boardChannels);

    Minimax survivor(W, H, M, N, P, boardChannels);

    survivor.updateRewards(survivor.findClosestConfig("best_params.txt"));

    while (nbPlayerAlive > 1 && Players[P].second.alive) {
        // Read and process moves from all players in the current round
        for (int playerTurn = 1; playerTurn <= N && nbPlayerAlive > 1 && Players[P].second.alive; playerTurn++)
        {
            if (playerTurn == P)
            {
                vector<float> playerBoardState = helper.computeBoardState(Players);

                // --- Select action using Minimax ---
                unordered_map<string, float> mmScores = survivor.calculateMoveScores(playerBoardState, Players, turn);

                // If no valid move, the snake will die so we need to train the network before killing it
                if (mmScores.empty())
                {
                    bestMove = "up";
                    Players[P].second.alive = false;
                    nbPlayerAlive--;
                }
                else
                {
                    // Select best combined move
                    bestMove = max_element(mmScores.begin(), mmScores.end(),
                    [](const pair<string, float> &a, const pair<string, float> &b) { return a.second < b.second; })->first;
                }

                Players = snakeUpdate(Players, P, M, round, bestMove);

                cout << bestMove << endl;
            }
            else
            {
                // --- Process opponents' moves ---
                cin >> action >> playerNb;
                if (action == "move")
                {
                    cin >> move;
                    Players = snakeUpdate(Players, playerNb+1, M, round, move);

                }
                else if (action == "death" && Players[playerNb+1].second.alive)
                {
                    cin >> action >> playerNb >> move;

                    Players[playerNb].second.alive = false;
                    nbPlayerAlive--;
                }
            }

            turn++;
        }
        round++;

        // Flush output buffer
        cout.flush();
    }
}


void trainingMain() {
    int W, H, M, N, P, X, Y;
    cin >> W >> H;  // size of the board
    cin >> M;       // grow 1 block every M rounds
    cin >> N >> P;  // number of players and player's turn


    // Store player nb (int), parts of their body (deque<coord> and data like
    // the origin of the snake, the accessible space and their risk of being blocked
    PlayersData Players;
    for (int i = 1; i <= N; i++) {
        cin >> X >> Y;
        Players[i].first.emplace_front(X, Y);  // push the head of the player

        // Initialise Player's data
        Players[i].second.origin = make_pair(X, Y);  // origin of the player
        Players[i].second.alive = true;
    }


    // Game loop variables
    string action;
    int playerNb;
    string move;
    int round = 0;
    int turn = 0;
    int nbPlayerAlive = N;
    string bestMove;

    deque<string> last5Moves;
    Transition transition;


    int boardChannels = 6; // 6 Channels for each case of the board

    // helper class
    Utils helper(W, H, M, N, P, boardChannels);

    // NN setup
    // Input dimension:
    const int CNN_OUTPUT_SIZE = 16;
    const int EXTRA_FEATURES_SIZE = 6*(N-1) + 5 + 2*(N-1);
    const int FC_INPUT_SIZE = CNN_OUTPUT_SIZE + EXTRA_FEATURES_SIZE;

    // main and target network
    NNAI mainNetwork(W, H, M, N, P, boardChannels, 0.1, 0.01, 0.995);

    mainNetwork.aiCNN.addLayer("ReLU", 3, 1, 1, 8, boardChannels);
    mainNetwork.aiCNN.addLayer("ReLU", 3, 1, 1, CNN_OUTPUT_SIZE);

    mainNetwork.aiFCNN.addLayer("ReLU", 64, FC_INPUT_SIZE);
    mainNetwork.aiFCNN.addLayer("ReLU", 32);
    mainNetwork.aiFCNN.addLayer("", 4);  // output layer

    // Minimax setup
    Minimax survivor(W, H, M, N, P, boardChannels);

    survivor.updateRewards(survivor.findClosestConfig("best_params.txt"));

    // Merging setup
    float lambda = 0.7;  // change to 0.0 to train miniMax, 1.0 for full DQN and 0.7 for basic use

    // Training parameters
    int batchSize = 64;
    float learningRate = 0.01;  // start with 0.01 then 0.001 to increase precision
    float gamma = 0.99;
    int targetUpdateFreq = 1000;
    int trainingStep = 0;


    while (nbPlayerAlive > 1 && Players[P].second.alive) {
        vector<float> currentBoardState = helper.computeBoardState(Players);
        vector<float> currentExtraState = helper.computeExtraFeatures(currentBoardState, Players, {}, last5Moves, turn);
        bool terminal = false;
        bool oppDied = false;

        // Read and process moves from all players in the current round
        for (int playerTurn = 1; playerTurn <= N && nbPlayerAlive > 1 && Players[P].second.alive; playerTurn++)
        {
            if (playerTurn == P)
            {
                vector<float> playerBoardState = helper.computeBoardState(Players);
                vector<float> playerExtraState = helper.computeExtraFeatures(playerBoardState, Players, {}, last5Moves, turn);


                // --- Select action using DQN ---
                vector<float> qValues = mainNetwork.forwardPropagation(playerBoardState, playerExtraState);

                // --- Select action using Minimax ---
                unordered_map<string, float> mmScores = survivor.calculateMoveScores(playerBoardState, Players, turn);


                /// Combine scores with lambda weighting
                map<string, float> combined;
                for (const auto &[move, score] : mmScores) {
                    int qIdx = distance(LABEL_ACTIONS.begin(), find_if(LABEL_ACTIONS.begin(), LABEL_ACTIONS.end(),
                        [&move](const pair<int, string> &p) { return p.second == move; }));
                    combined[move] = (1 - lambda) * score + lambda * qValues[qIdx];

                    cout << "> MiniMax Score: " << score << " DQN Q-Value: " << qValues[qIdx] << " Combined: " << combined[move] << endl;
                }

                // If no valid move, the snake will die so we need to train the network before killing it
                if (combined.empty())
                {
                    bestMove = "up";

                    Players[P].second.alive = false;
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

                Players = snakeUpdate(Players, P, M, round, bestMove);

                // --- Compute next state ---
                vector<float> nextBoardState = helper.computeBoardState(Players);

                bool oppWillDie = true;
                int nbOppKillSetup = 0;
                unordered_map<int, vector<string>> playersValidMoves;
                for (auto &player : Players)
                {
                    playersValidMoves[player.first] = helper.getValidMoves(nextBoardState, Players, player.first);
                    if (playersValidMoves[player.first].size() <= 2)
                    {
                        nbOppKillSetup++;
                    }

                    if (player.first != P && oppWillDie && playersValidMoves[player.first].size() != 0)  // check if every other opponents will die next round
                    {
                        oppWillDie = false;
                    }
                }

                vector<float> nextExtraState = helper.computeExtraFeatures(nextBoardState, Players,
                    playersValidMoves, last5Moves, turn);

                terminal = helper.checkTerminalState(nextBoardState, Players) || oppWillDie;

                // --- Compute reward ---
                float reward = mainNetwork.computeReward(nextBoardState, nextExtraState,
                    currentExtraState, Players, P, turn, nbOppKillSetup,
                        oppWillDie, false);

                // --- Create transition ---
                transition = mainNetwork.replayBuffer.createTransition(currentBoardState, currentExtraState,
                                        nextBoardState, nextExtraState, bestMove, turn, Players, reward, terminal);

                if (terminal)
                {
                    // avoid training on the first rounds
                    if (round > 5)
                    {
                        cout << "> reward: " << transition.reward << endl;
                        cout << "> done: " << transition.done << endl;

                        // Send transition to the training server
                        send_transition(transition);
                    }

                    Players[P].second.alive = false;
                    nbPlayerAlive--;
                }

                cout << bestMove << endl;
            }
            else
            {
                // --- Process opponents' moves ---
                cin >> action >> playerNb;
                if (action == "move")
                {
                    cin >> move;
                    Players = snakeUpdate(Players, playerNb+1, M, round, move);

                }
                else if (action == "death" && Players[playerNb+1].second.alive)
                {
                    cin >> action >> playerNb >> move;

                    // Mark the specified player as dead
                    Players[playerNb].second.alive = false;
                    nbPlayerAlive--;
                }
            }

            turn++;
        }

        if (!terminal && Players[P].second.alive)
        {
            vector<float> nextBoardState = helper.computeBoardState(Players);
            vector<float> nextExtraState = helper.computeExtraFeatures(nextBoardState, Players, {},
                                                                        last5Moves, turn);

            int nbOppKillSetup = 0;
            for (auto &player : Players)
            {
                int nbOppValidMoves = helper.getValidMoves(nextBoardState, Players, player.first).size();
                if (nbOppValidMoves <= 2)
                {
                    nbOppKillSetup++;
                }
            }

            // --- Compute reward ---
            float reward = mainNetwork.computeReward(nextBoardState, nextExtraState, currentExtraState,
                            Players, P, turn, nbOppKillSetup, false, oppDied);

            transition = mainNetwork.replayBuffer.createTransition(currentBoardState, currentExtraState, nextBoardState,
                                                                   nextExtraState, bestMove, turn, Players, reward, terminal);


        }

        if (!terminal)
        {
            if (round > 5)
            {
                cout << "> reward: " << transition.reward << endl;
                cout << "> done: " << transition.done << endl;

                send_transition(transition);
            }
        }

        round++;

        // Flush output buffer
        cout.flush();
    }
}


void gameMain() {
    int W, H, M, N, P, X, Y;
    cin >> W >> H;  // size of the board
    cin >> M;       // grow 1 block every M rounds
    cin >> N >> P;  // number of players and player's turn


    // Store player nb (int), parts of their body (deque<coord> and data like
    // the origin of the snake, the accessible space and their risk of being blocked
    PlayersData Players;
    for (int i = 1; i <= N; i++) {
        cin >> X >> Y;
        Players[i].first.emplace_front(X, Y);  // push the head of the player

        // Initialise Player's data
        Players[i].second.origin = make_pair(X, Y);  // origin of the player
        Players[i].second.alive = true;
    }


    // Game loop variables
    string action;
    int playerNb;
    string move;
    int round = 0;
    int turn = 0;
    int nbPlayerAlive = N;
    string bestMove;
    deque<string> last5Moves;


    int boardChannels = 6; // 6 Channels for each case of the board

    // helper class
    Utils helper(W, H, M, N, P, boardChannels);

    // NN setup

    // main and target network
    NNAI mainNetwork(W, H, M, N, P, boardChannels, 0.1, 0.01, 0.995);
    mainNetwork.loadModelFromFile("model.model");

    // Minimax setup
    Minimax survivor(W, H, M, N, P, boardChannels);

    survivor.updateRewards(survivor.findClosestConfig("best_params.txt"));

    // Merging setup
    float lambda = 0.7;  // change to 0.0 to train miniMax, 1.0 for full DQN and 0.7 for basic use


    while (nbPlayerAlive > 1 && Players[P].second.alive) {
        // Read and process moves from all players in the current round
        for (int playerTurn = 1; playerTurn <= N && nbPlayerAlive > 1 && Players[P].second.alive; playerTurn++)
        {
            if (playerTurn == P)
            {
                vector<float> playerBoardState = helper.computeBoardState(Players);
                vector<float> playerExtraState = helper.computeExtraFeatures(playerBoardState, Players, {}, last5Moves, turn);


                // --- Select action using DQN ---
                vector<float> qValues = mainNetwork.forwardPropagation(playerBoardState, playerExtraState);

                // --- Select action using Minimax ---
                unordered_map<string, float> mmScores = survivor.calculateMoveScores(playerBoardState, Players, turn);

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
                    Players[P].second.alive = false;
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

                Players = snakeUpdate(Players, P, M, round, bestMove);

                cout << bestMove << endl;
            }
            else
            {
                // --- Process opponents' moves ---
                cin >> action >> playerNb;
                if (action == "move")
                {
                    cin >> move;
                    Players = snakeUpdate(Players, playerNb+1, M, round, move);

                }
                else if (action == "death" && Players[playerNb+1].second.alive)
                {
                    cin >> action >> playerNb >> move;

                    // Mark the specified player as dead
                    Players[playerNb].second.alive = false;
                    nbPlayerAlive--;
                }
            }

            turn++;
        }
        round++;

        // Flush output buffer
        cout.flush();
    }
}


int main() {
    gameMain();
    return 0;
}