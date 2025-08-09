/*************************************************************************
StateAnalyzer - todo
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//- Implementation of the module <StateAnalyzer> (file StateAnalyzer.cpp) -

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <iostream>

using namespace std;

//------------------------------------------------------- Personal Include
#include "StateAnalyzer.h"


//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
vector<float> StateAnalyzer::computeExtraFeatures(const vector<float> boardState, const Players &playerState,
                    unordered_map<int, vector<string>> validMoves, const deque<string> &last5Moves, int turn)
// Algorithm : Compute extra features for the DQN model
// Spatial metrics :
//     For every opponent: min distance between my snake's head and the opponent's body (normalized by max distance)
//     For every opponent: min distance between the opponent's head and my body (normalized by max distance)
// growth and size metrics :
//     growth speed (normalized by the max growth speed) : M/10
//     growth progress (normalized by M) : (M - (currentRound % M))/M
//     snake size (normalized by the board size) : snake.size() / (W * H)
// game metrics :
//     game progress (normalized by the max number of turns) : min(turn / (MAX_TURNS), 1.0)
// gameplay and aggressivity metrics :
//     action History Entropy (normalized between -1 and 1) : tanh(calculateActionEntropy(last5Moves))
//     For every opponent: escape routes (normalized 0-1)
//     For every opponent: kill zone score (0-1)
{
    vector<float> extras;

    if (validMoves.empty())
    {
        for (auto &player : playerState)
        {
            if (player.first != gameEngine.config.P)
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

            botToPlayer.min = playerToBot.min = MAX_DISTANCE;  // max distance allowed
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

float Reward::calculateKillZoneProximity(Coord oppHead) const
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


float StateAnalyzer::calculateActionEntropy(const deque<string>& moves)
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
