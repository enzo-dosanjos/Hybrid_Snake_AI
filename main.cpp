#include <iostream>
#include <fstream>
#include <unordered_map>
#include <map>
#include <utility>
#include <deque>
#include <vector>
#include <cmath>

using namespace std;


struct inputCanals
{
    bool botHead;
    bool botBody;
    bool oppHead;
    bool oppBody;

    inputCanals(bool bH = false, bool bB = false, bool oH = false, bool oB = false)
                : botHead(bH), botBody(bB), oppHead(oH), oppBody(oB) {}
};

struct inputMaxMinAvg
{
    int max;
    int min;
    int avg;
    inputMaxMinAvg() : max(0), min(1000000), avg(0) {} // Valeurs d'initialisation (hard coded mais bon, ça devrait suffir)
};


vector<double> multiplyMatrix(const vector<double>& row, const vector<vector<double>>& matrix) {
    const int M = matrix[0].size();
    vector<double> result(M, 0.0);

    // Optimized multiplication
    for (int i = 0; i < row.size(); ++i)
    {
        const double row_val = row[i];
        for (int j = 0; j < M; ++j)
        {
            result[j] += row_val * matrix[i][j];
        }
    }

    return result;
}


int main() {
    int W, H, M, N, P, X, Y;
    cin >> W >> H;  // size of the board
    cin >> M;       // grow 1 block every M rounds
    cin >> N >> P;  // number of players and player's turn

    cout << "> Received grid size: " << W << "x" << H << endl;
    cout << "> Growth rate: every " << M << " turns" << endl;
    cout << "> Number of players: " << N << " (I am player " << P << ")" << endl;

    typedef pair<int, int> coord;

    unordered_map<int, deque<coord>> Players;
    for (int i = 1; i <= N; i++) {
        cin >> X >> Y;
        Players[i].push_front(make_pair(X, Y));
        cout << "> Player " << i << " starts at (" << X << "," << Y << ")" << endl;
    }

    map<string, coord> directions = {{"left", make_pair(-1, 0)}, {"right", make_pair(1, 0)},
                                      {"up", make_pair(0, -1)}, {"down", make_pair(0, 1)}};

    const string directionsCycle[] = {"right", "down", "left", "up"};
    int dir_index = 0;

    // Game loop variables
    string move;
    string trash;
    int moveLen;
    int round = 0;


    while (true) {
        vector<inputCanals> board(H * W, inputCanals());

        vector<float> inputs;

        // Read and process moves from all previous players in the current round
        for (int playerTurn = 1; playerTurn <= N; ++playerTurn)
        {
            if (playerTurn == P)
            {
                // Determine and send current player's move
                string chosenDir = directionsCycle[dir_index];
                cout << "> Choosing direction: " << chosenDir << endl;
                cout << chosenDir << endl;
                dir_index = (dir_index + 1) % 4;

                // Update current player's snake
                coord currentHead = Players[P].front();
                Players[P].push_front(make_pair(currentHead.first + directions[chosenDir].first,
                                                currentHead.second + directions[chosenDir].second
                                            ));
            }
            else 
            {
                cin >> trash >> moveLen >> move;
                cout << "> Received playerTurn " << playerTurn << "'s move: " << move << endl;

                // Update playerTurn's snake
                coord currentHead = Players[playerTurn].front();
                Players[playerTurn].push_front(make_pair(currentHead.first + directions[move].first,
                                                    currentHead.second + directions[move].second
                                                    ));
            }

            if (round % M != 0)
            {
                Players[playerTurn].pop_back();
            }

            for (int i = 0; i < Players[playerTurn].size(); i++)
            {
                cout << "> Player " << playerTurn <<"'s Snake part " << i << ": " << Players[playerTurn][i].first << ", " << Players[playerTurn][i].second << endl;
            }
        }

        for (const auto& player : Players)
        {
            int playerNb = player.first;
            const deque<coord>& snake = player.second;

            inputMaxMinAvg playerToBot;

            for (int i = 0; i < snake.size(); i++)
            {
                int x = snake[i].first;
                int y = snake[i].second;
                int index = y * W + x; // 2D coords to index

                if (playerNb == P)
                {
                    if (i == 0)
                    {
                        board[index].botHead = true; // Head of the bot's snake
                    }
                    else
                    {
                        board[index].botBody = true; // Body of the bot's snake
                    }
                }
                else
                {
                    int botHeadX = Players[P][0].first;
                    int botHeadY = Players[P][0].second;
                    float dist = sqrt(pow(abs(x-botHeadX), 2) + pow(abs(y-botHeadY), 2));

                    playerToBot.avg += dist;

                    if (dist > playerToBot.max)
                    {
                        playerToBot.max = dist;
                    }

                    if (dist < playerToBot.min)
                    {
                        playerToBot.min = dist;
                    }

                    if (i == 0)
                    {
                        board[index].oppHead = true; // Head of an opponent's snake

                        inputMaxMinAvg botToPlayer;
                        botToPlayer.max = botToPlayer.min = sqrt(pow(W, 2) + pow(H, 2));
                        botToPlayer.avg = 0;
                        for (int j = 0; j < snake.size(); j++)
                        {
                            int botX = Players[P][j].first;
                            int botY = Players[P][j].second;
                            dist = sqrt(pow(abs(x - botX), 2) + pow(abs(y - botY), 2));
                            botToPlayer.avg += dist;

                            if (dist > botToPlayer.max)
                            {
                                botToPlayer.max = dist;
                            }

                            if (dist < botToPlayer.min)
                            {
                                botToPlayer.min = dist;
                            }
                        }
                        
                        botToPlayer.avg /= snake.size();

                        inputs.push_back(botToPlayer.min);
                        inputs.push_back(botToPlayer.max);
                        inputs.push_back(botToPlayer.avg);

                    }
                    else
                    {
                        board[index].oppBody = true; // Body of an opponent's snake
                    }

                }
            }

            playerToBot.avg /= snake.size();

            inputs.push_back(playerToBot.min);
            inputs.push_back(playerToBot.max);
            inputs.push_back(playerToBot.avg);

        }

        // Print the board for debugging
        cout << "> Board state:" << endl;
        for (int y = 0; y < H; ++y)
        {
            cout << "> ";
            for (int x = 0; x < W; ++x)
            {
                int index = y * W + x;
                cout << "(" << board[index].botHead << board[index].botBody << board[index].oppHead << board[index].oppBody << ") ";
            }
            cout << endl;
        }

        // Last Inputs : growth and snake size
        inputs.push_back(M);
        inputs.push_back(Players[1].size());

        round++;

        cout << "> Snake Size: " << Players[1].size() << endl;
        cout << "> round: " << round << endl;

        // Flush output buffer
        cout.flush();
    }

    return 0;
}