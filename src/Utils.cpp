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
bool Utils::isInbound(int x, int y)
// Algorithm : Check if the coordinates are in the board's bounds
{
    return x >= 0 && x < config.W && y >= 0 && y < config.H;
}  //----- End of isInbound


bool Utils::checkDeath(const PlayersData &playerState, const vector<float> &boardState, int playerNb)
// Algorithm : Check if the player is dead
{
    // todo: check for the moment when opponent's die face to face by checking if their body is next to their head (not a problem in 1v1)

    Coord playerHead = playerState.at(playerNb).first.front();

    // check if the move is inbounds
    if (isInbound(playerHead.first, playerHead.second))
    {
        // Check if the cell is empty (no part of my body or no other player)
        bool cellEmpty = true;
        int ind = (playerHead.second * config.W + playerHead.first) * boardChannels;
        for (int i = 0; i < boardChannels; i++)
        {
            if ((i != 0 || playerNb != config.P) && (i != 2 || playerNb == config.P))  // if playerNb is P, at i == 0 is
                                                                                    // its head, same for opp and i == 2
            {
                if (boardState[ind + i] != 0.0)
                {
                    cellEmpty = false;
                }
            }
        }

        if (cellEmpty)
        {
            return false;
        }
    }

    return true;
}  //----- End of checkDeath


bool Utils::checkTerminalState(vector<float> boardState, PlayersData playerState)
// Algorithm : Check if the game is over
{
    int alivePlayersNb = 0;
    bool playerAlive = false;  // check if my player is alive

    for (const auto &player : playerState)
    {
        if (!checkDeath(playerState, boardState, player.first))
        {
            alivePlayersNb++;
            if (player.first == config.P)
            {
                playerAlive = true;
            }
        }
    }

    return !playerAlive || alivePlayersNb <= 1;
}  // ----- End of checkTerminalState


vector<string> Utils::getValidMoves(const vector<float> &boardState, const PlayersData &playerState, int playerNb)
// Algorithm : Get the valid moves for the player
{
    vector<string> validMoves;

    // skip dead players
    if (!playerState.at(playerNb).second.alive)
    {
        return {};
    }

    Coord currentHead = playerState.at(playerNb).first.front();
    for (auto direction : DIRECTIONS)
    {
        Coord newHead = make_pair(currentHead.first + direction.second.first,
                        currentHead.second + direction.second.second
                        );

        // check if the move is inbounds
        if (isInbound(newHead.first, newHead.second))
        {
            // Check if the cell is empty (no part of my body or no other player)
            bool cellEmpty = true;
            int ind = (newHead.second * config.W + newHead.first) * boardChannels;

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


int Utils::getNextPlayer(int currentPlayer, const PlayersData &playerState)
// Algorithm : Get the next player
{
    int next = (currentPlayer % config.N) + 1;
    while (next != currentPlayer)
    {
        if (playerState.at(next).second.alive)
        {
            return next;
        }
        next = (next % config.N) + 1;
    }

    return -1; // Every other player are dead
}  // ----- End of getNextPlayer


vector<float> Utils::computeBoardState(const PlayersData &Players)
// Algorithm : Compute the board state
// board state is a flattened vector<float> with 6 channels per cell:
// 0: my snake head, 1: my snake body, 2: opponent head, 3: opponent body
// 4: x coordinate of the cell, 5: y coordinate of the cell
{
    vector<float> board(config.W * config.H * boardChannels, 0.0);

    for (const auto &player : Players)
    {
        int playerNb = player.first;
        const deque<Coord> &snake = player.second.first;

        for (size_t i = 0; i < snake.size(); i++)
        {
            int x = snake[i].first;
            int y = snake[i].second;
            int ind = (y * config.W + x) * boardChannels;  // index of the cell at x, y

            if (playerNb == config.P)
            {
                if (i == 0)
                {
                    board[ind] = 1.0;      // my snake head
                }
                else
                {
                    board[ind + 1] = 1.0;  // my snake body
                }
            }
            else
            {
                if (i == 0)
                {
                    board[ind + 2] = 1.0;  // opponent head
                }
                else
                {
                    board[ind + 3] = 1.0;  // opponent body
                }
            }
        }
    }
    return board;
}  // ----- End of computeBoardState


vector<float> Utils::computeExtraFeatures(const vector<float> boardState, const PlayersData &playerState,
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
        if (player.second.second.alive)
        {
            nbAlivePlayers++;
        }
        if (player.first != config.P)
        {
            // Opponent Escape Route Counter
            escapeRoutes.push_back(validMoves[player.first].size() / 4.0);  // Normalized 0-1

            // Kill Zone Proximity (0-1)
            killZoneScore.push_back(calculateKillZoneProximity(player.second.first.front()));
        }
    }

    // Phase-Based Aggression Coefficient
    const float MAX_TURNS = (config.M > 1) ? 0.8*(config.W * config.H * config.M) : (config.W * config.H * config.M);
    float phase = min(((float)turn / MAX_TURNS), 1.0f);  // Assuming W*H*M turn max for the game

    // action History Entropy (measure passivity)
    float entropy = tanh(calculateActionEntropy(last5Moves));  // normalised between -1 and 1


    InputMinMaxAvg playerToBot;
    InputMinMaxAvg botToPlayer;

    Coord myHead = playerState.at(config.P).first.front();
    int mySnakeSize = playerState.at(config.P).first.size();

    const float MAX_DISTANCE = sqrt(config.W * config.W + config.H * config.H);

    for (const auto &player : playerState)
    {
        if (player.first != config.P)
        {
            const deque<Coord> &snake = player.second.first;

            botToPlayer.min = playerToBot.min = sqrt(config.W * config.W + config.H * config.H);  // max distance allowed
                                                                                                  // by the board's size
            botToPlayer.avg = playerToBot.avg = botToPlayer.max = playerToBot.max = 0;

            for (size_t i = 0; i < snake.size(); i++)
            {
                float x = snake[i].first;
                float y = snake[i].second;

                float dist = sqrt((x-myHead.first)*(x-myHead.first) + (y-myHead.second)*(y-myHead.second));

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
                        float myX = playerState.at(config.P).first[j].first;
                        float myY = playerState.at(config.P).first[j].second;
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
        (float)playerState.at(config.P).first.size() / (float)(config.W * config.H)  // normalise between 0 and 1
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


float Utils::calculateKillZoneProximity(Coord oppHead)
// Algorithm : Calculate the proximity of a player to a kill zone (corners and walls)
{
    // Calculate corner proximity
    float cornerWeight = 0.0;
    const float quartW = config.W/4.0;
    const float quartH = config.H/4.0;

    // Check if in outer 25% of board edges
    if(oppHead.first < quartW || oppHead.first > config.W - quartW)
    {
        cornerWeight += 0.5;
    }

    if(oppHead.second < quartH || oppHead.second > config.H - quartH)
    {
        cornerWeight += 0.5;
    }

    // Calculate the minimum distance to any wall
    float wallDist = min({
        (float)oppHead.first,
        (float)(config.W-1 - oppHead.first),
        (float)oppHead.second,
        (float)(config.H-1 - oppHead.second)
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