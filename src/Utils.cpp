/*************************************************************************
Utils - utility functions for
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//-------- Implementation of the module <Utils> (file Utils.cpp) ---------

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace std;

//------------------------------------------------------- Personal Include
#include "../include/Utils.h"


//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
vector<string> Utils::getValidMoves(const vector<float> &boardState, const Players &playerState, int playerNum)
// Algorithm : Get the valid moves for the player
{
    vector<string> validMoves;

    // skip dead players
    if (!playerState.at(playerNum).playerInfo.alive)
    {
        return {};
    }

    Coord currentHead = playerState.at(playerNum).snake.front();
    for (auto direction : DIRECTIONS)
    {
        Coord newHead = {currentHead.x + direction.second.x,
                        currentHead.y + direction.second.y
                        };

        // check if the move is inbounds
        if (isInbound(newHead.x, newHead.y))
        {
            // Check if the cell is empty (no part of my body or no other player)
            bool cellEmpty = true;
            int ind = (newHead.y * config.W + newHead.x) * boardChannels;

            for (int i = 0; i < boardChannels; i++)
            {
                if (boardState[ind + i] != 0.0)
                {
                    cellEmpty = false;
                    break;
                }
            }

            if (cellEmpty)
            {
                validMoves.push_back(direction.first);
            }
        }
    }

    return validMoves;
}  // ----- End of getValidMoves

vector<float> Utils::computeExtraFeatures(const vector<float> boardState, const Players &playerState,
                    unordered_map<int, vector<string>> validMoves, const deque<string> &last5Moves, int turn)
// Algorithm : Compute extra features for the DQN model
// Features:
// For every opponent: min, max, avg distances between my snake's head and the opponent's body (normalized by max distance)
// For every opponent: min, max, avg distances between the opponent's head and my body (normalized by max distance)
// growth speed
// nb of round before growth
// snake size
// For every opponent: escape routes (normalized 0-1)
// For every opponent: kill zone score (0-1)
// Phase-Based Aggression Coefficient (0-1)
// action History Entropy (-1-1)
{
    vector<float> extras;

    if (validMoves.empty())
    {
        for (auto &player : playerState)
        {
            if (player.first != config.P)
            {
                validMoves[player.first] = getValidMoves(boardState, playerState, player.first);
            }
        }
    }

    int nbAlivePlayers = 0;
    vector<float> escapeRoutes;
    vector<float> killZoneScore;
    for (auto &player : playerState)
    {
        if (player.second.playerInfo.alive)
        {
            nbAlivePlayers++;
        }
        if (player.first != config.P)
        {
            // Opponent Escape Route Counter
            escapeRoutes.push_back(validMoves[player.first].size() / 4.0);  // Normalized 0-1

            // Kill Zone Proximity (0-1)
            killZoneScore.push_back(calculateKillZoneProximity(player.second.snake.front()));
        }
    }

    // Phase-Based Aggression Coefficient
    const float MAX_TURNS = (config.M > 1) ? 0.8*(config.W * config.H * config.M) : (config.W * config.H * config.M);
    float phase = min(((float)turn / MAX_TURNS), 1.0f);  // Assuming W*H*M turn max for the game

    // action History Entropy (measure passivity)
    float entropy = tanh(calculateActionEntropy(last5Moves));  // normalised between -1 and 1


    InputMinMaxAvg playerToBot;
    InputMinMaxAvg botToPlayer;

    Coord myHead = playerState.at(config.P).snake.front();
    int mySnakeSize = playerState.at(config.P).snake.size();

    const float MAX_DISTANCE = sqrt(config.W * config.W + config.H * config.H);

    for (const auto &player : playerState)
    {
        if (player.first != config.P)
        {
            const deque<Coord> &snake = player.second.snake;

            botToPlayer.min = playerToBot.min = sqrt(config.W * config.W + config.H * config.H);  // max distance allowed
                                                                                                  // by the board's size
            botToPlayer.avg = playerToBot.avg = botToPlayer.max = playerToBot.max = 0;

            for (size_t i = 0; i < snake.size(); i++)
            {
                float x = snake[i].x;
                float y = snake[i].y;

                float dist = sqrt((x-myHead.x)*(x-myHead.x) + (y-myHead.y)*(y-myHead.y));

                // Normalise every distances
                dist /= MAX_DISTANCE;

                // Compute Min, Max and Avg between my Head to the opponent's body
                playerToBot.avg += dist/snake.size();

                playerToBot.max = max(playerToBot.max, dist);

                playerToBot.min = min(playerToBot.min, dist);

                if (i == 0)
                {
                    for (int j = 0; j < mySnakeSize; j++)
                    {
                        float myX = playerState.at(config.P).snake[j].x;
                        float myY = playerState.at(config.P).snake[j].y;
                        dist = sqrt((x - myX)*(x - myX) + (y - myY)*(y - myY));

                        // Normalise every distances
                        dist /= MAX_DISTANCE;

                        // Compute Min, Max and Avg between the opponent's head to my body
                        botToPlayer.avg += dist/mySnakeSize;

                        botToPlayer.min = min(botToPlayer.min, dist);

                        botToPlayer.max = max(botToPlayer.max, dist);
                    }
                }
            }

            extras.insert(extras.end(),
            {
                playerToBot.min,
                playerToBot.max,
                playerToBot.avg,
                botToPlayer.min,
                botToPlayer.max,
                botToPlayer.avg
            });
        }
    }

    // growth speed, nb of round before growth and snake size
    int currentRound = ((turn) / nbAlivePlayers);
    extras.insert(extras.end(),
    {
        (float)config.M/10.0f,  // normalise M between 0 and 1
        (float)(config.M - (currentRound % config.M))/(float)config.M,    // normalise between 0 and 1
        (float)playerState.at(config.P).snake.size() / (float)(config.W * config.H)  // normalise between 0 and 1
    });


    for (size_t i = 0; i < escapeRoutes.size(); i++)
    {
        extras.insert(extras.end(), {
            escapeRoutes[i],
            killZoneScore[i]
        });
    }

    extras.insert(extras.end(), {
        phase,
        entropy
    });

    return extras;
}  // ----- End of computeExtraFeatures


float Utils::calculateKillZoneProximity(Coord oppHead) const
// Algorithm : Calculate the proximity of a player to a kill zone (corners and walls)
{
    // Calculate corner proximity
    float cornerWeight = 0.0;
    const float quartW = config.W/4.0;
    const float quartH = config.H/4.0;

    // Check if in outer 25% of board edges
    if(oppHead.x < quartW || oppHead.x > config.W - quartW)
    {
        cornerWeight += 0.5;
    }

    if(oppHead.y < quartH || oppHead.y > config.H - quartH)
    {
        cornerWeight += 0.5;
    }

    // Calculate the minimum distance to any wall
    float wallDist = min({
        (float)oppHead.x,
        (float)(config.W-1 - oppHead.x),
        (float)oppHead.y,
        (float)(config.H-1 - oppHead.y)
    });

    // Normalize and invert (0 = at wall, 1 = center)
    float normalizedWall = 1.0 - (wallDist / (max(config.W, config.H)/2.0));

    // Combine factors (corner presence + wall proximity)
    return (cornerWeight + normalizedWall) / 2.0;
}  // ----- End of calculateKillZoneProximity


float Utils::calculateActionEntropy(const deque<string>& moves)
// Algorithm : Calculate the entropy of the action history
{
    if (moves.empty())
    {
        return 0.0; // Max entropy when no history
    }

    map<string, int> counts;
    for(const auto& m : moves)
    {
        counts[m]++;
    }

    float entropy = 0.0;
    for(const auto &[move, cnt] : counts)
    {
        float p = cnt / (float)moves.size();
        entropy -= p * log2(p);
    }
    return entropy / 2.0;  // Max entropy for 4 moves = 2.0
}  // ----- End of calculateActionEntropy


Matrix<vector<float>> Utils::reshapeBoardState(const vector<float> &flatBoardState) const
// Algorithm : Reshape the flat board state into a 3D matrix of the shape [channels][H][W]
{
    // Allocate the shaped board
    Matrix<vector<float>> reshapedBoard(boardChannels, config.W, vector<float>(config.H, 0.0));

    // Fill the shaped structure
    for (int c = 0; c < boardChannels; c++)
    {
        for (int h = 0; h < config.H; h++)
        {
            for (int w = 0; w < config.W; w++)
            {
                int index = (h * config.W + w) * boardChannels + c;
                reshapedBoard[c][h][w] = flatBoardState[index];
            }
        }
    }

    return reshapedBoard;
}