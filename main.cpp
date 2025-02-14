#include <iostream>
#include <fstream>
#include <unordered_map>
#include <map>
#include <utility>
#include <deque>
#include <vector>
#include <cmath>

using namespace std;


struct inputMinMaxAvg
{
    float min;
    float max;
    float avg;
};


struct Layer {
    string type;                // activation function (ex: ReLU)
    int input_size;
    int output_size;

    float *weights;
    float *biases;
    float *output;

    float *weight_gradient;
    float *bias_gradient;
    float *error;
};


class NNAI
{
    public:
        void ReLU(float *vals, int len) {
            for (int i = 0; i < len; i++) {
                if (vals[i] < 0) {
                    vals[i] = 0;
                }
            }
        } //----- end of ReLU


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
        } //----- end of multiplyMatrix


        void initLayer (float *layer_weight, float *layer_bias, int input_size, int layer_size) {
            // Initialize the random number generator
            srand(static_cast<unsigned int>(time(0)));
        
            // initialise weight array with random values
            for (int i = 0; i < input_size * layer_size; i++) {
                layer_weight[i] = static_cast<float>(rand()) / RAND_MAX * 0.2f - 0.1f; // Uniform distribution between -0.1 and 0.1;
        
                if (i < layer_size) {
                    layer_bias[i] = 0; // Initialise biases to 0
                }
            }
        } //----- end of initLayer


        // Constructors and Destructors
        NNAI () {

        }

        ~NNAI () {

        }

    protected:
        vector<Layer> NN;
};


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
        vector<bool> board(H * W * 4, false);  // 4 Canals for each case of the board
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

            inputMinMaxAvg playerToBot;
            inputMinMaxAvg botToPlayer;
            botToPlayer.min = playerToBot.min = sqrt(pow(W, 2) + pow(H, 2));
            botToPlayer.avg = playerToBot.avg = botToPlayer.max = playerToBot.max = 0;

            for (int i = 0; i < snake.size(); i++)
            {
                float x = snake[i].first;
                float y = snake[i].second;
                int index = (y * W + x) * 4; // 2D coords to index

                if (playerNb == P)
                {
                    if (i == 0)
                    {
                        board[index] = true; // Head of the bot's snake
                    }
                    else
                    {
                        board[index + 1] = true; // Body of the bot's snake
                    }
                }
                else
                {
                    float botHeadX = Players[P][0].first;
                    float botHeadY = Players[P][0].second;
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
                        board[index + 2] = true; // Head of an opponent's snake

                        for (int j = 0; j < snake.size(); j++)
                        {
                            float botX = Players[P][j].first;
                            float botY = Players[P][j].second;
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
                    }
                    else
                    {
                        board[index + 3] = true; // Body of an opponent's snake
                    }

                }
            }

            if (playerNb != P)
            {
                playerToBot.avg /= snake.size();

                inputs.push_back(playerToBot.min);
                inputs.push_back(playerToBot.max);
                inputs.push_back(playerToBot.avg);

                inputs.push_back(botToPlayer.min);
                inputs.push_back(botToPlayer.max);
                inputs.push_back(botToPlayer.avg);
            }

        }

        // Print the board for debugging
        cout << "> Board state:" << endl;
        for (int y = 0; y < H; ++y)
        {
            cout << "> ";
            for (int x = 0; x < W; ++x)
            {
                int index = (y * W + x) * 4;
                cout << "(" << board[index] << board[index + 1] << board[index + 2] << board[index + 3] << ") ";
            }
            cout << endl;
        }

        // Last Inputs : growth and snake size
        inputs.push_back(M);
        inputs.push_back(Players[1].size());

        cout << "> ";
        for (auto input : inputs)
        {
            cout << input << ", ";
        }
        cout << endl;

        round++;

        cout << "> Snake Size: " << Players[1].size() << endl;
        cout << "> round: " << round << endl;

        // Flush output buffer
        cout.flush();
    }

    return 0;
}