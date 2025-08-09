/*************************************************************************
Minimax - Implementation of a minimax algorithm with alpha-beta prunning
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//------ Implementation of the module <Minimax> (file Minimax.cpp) ------

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>
#include <algorithm>

using namespace std;

//------------------------------------------------------- Personal Include
#include "../include/Minimax.h"

//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
unordered_map<string, float> Minimax::calculateMoveScores(vector<float> &boardState,
                                                          Players &playerState, int turn)
// Algorithm : Main method to calculate the move scores and return the best move.
// It uses the minimax algorithm with alpha-beta prunning to reduce the search space.
{
    unordered_map<string, float> moveScores;

    // Get valid moves for the current player
    vector<string> validMoves = helper.getValidMoves(boardState, playerState, config.P);  // todo: modify

    // Initialize alpha and beta values
    float alpha = -numeric_limits<float>::infinity();
    float beta = numeric_limits<float>::infinity();
    float bestValue = -numeric_limits<float>::infinity();

    for (const auto &move : validMoves) {
        // Simulate the move
        auto simulatedState = simulateMove(move, boardState, playerState, turn, config.P);
        int nextPlayer = helper.getNextPlayer(config.P, simulatedState.second);  // todo: modify

        float eval;
        if (nextPlayer == -1)
        {
            // if a terminal state is reached, evaluate the state
            eval = evaluateState(simulatedState.second, config.P);
        }
        else
        {
            // else, run the minimax algorithm
            eval = minimax(simulatedState.first, simulatedState.second, params.depth - 1, alpha, beta, nextPlayer, turn + 1);
        }

        // Update the best value, alpha and beta
        bestValue = max(bestValue, eval);
        alpha = max(alpha, bestValue);

        if (beta <= alpha) break; // Prunning

        moveScores[move] = eval;
    }

    return moveScores;
}  // ----- End of calculateMoveScores

//-------------------------------------------------------------- PROTECTED
pair<vector<float>, Players> Minimax::simulateMove(const string &move, const vector<float> &boardState,  // todo:change boardState shape
                                                       const Players &playerState, int turn, int playerNum)
// Algorithm : Simulate the move of a player and update the board and player states.
{
    // Make a copy of the current state
    pair<vector<float>, Players> newState;
    newState.first = boardState;
    newState.second = playerState;


    // Get current head and tail positions
    Coord currentHead = newState.second[playerNum].snake.front();
    Coord currentTail = newState.second[playerNum].snake.back();

    // create new head position from the move
    Coord newHead = {currentHead.x + DIRECTIONS.at(move).x,
                     currentHead.y + DIRECTIONS.at(move).y
    };


    // Update snake based on chosen move
    newState.second[playerNum].snake.push_front(newHead);

    // Count alive players after the move
    int alivePlayers = 0;
    for (const auto &player : newState.second)
    {
        if (!gameEngine.checkDeath(newState.second, newState.first, player.first))
        {
            alivePlayers++;
        }
        else
        {
            newState.second[player.first].playerInfo.alive = false;
        }
    }

    int currentRound = ((turn) / alivePlayers); // turn+1 because this move increments the turn


    // Remove tail segment if it's not a growth round from player states
    if (currentRound % config.M != 0)
    {
        newState.second[playerNum].snake.pop_back();
    }


    int ind = (currentHead.y * config.W + currentHead.x) * boardChannels;  // current head
    int newInd = (newHead.y * config.W + newHead.x) * boardChannels;  // new head

    // Make the current head become a body part and create the new head
    if (playerNum == config.P)
    {
        if (newState.first[ind] == 1.0)
        {
            newState.first[ind] = 0.0;
            newState.first[ind + 1] = 1.0;
        }
        newState.first[newInd] = 1.0;

        // remove tail segment if it's not a growth round from board state
        if (currentRound % config.M != 0)
        {
            newState.first[(currentTail.y * config.W + currentTail.x) * boardChannels + 1] = 0.0;
        }
    }
    else
    {
        if (newState.first[ind + 2] == 1.0)
        {
            newState.first[ind + 2] = 0.0;
            newState.first[ind + 3] = 1.0;
        }
        newState.first[newInd + 2] = 1.0;

        if (currentRound % config.M != 0)
        {
            newState.first[(currentTail.y * config.W + currentTail.x) * boardChannels + 3] = 0.0;
        }
    }

    return newState;
}  // ----- End of simulateMove


float Minimax::evaluateState(const Players &playerState, int currentPlayer)
// Algorithm : Evaluate the state of the game and return a score based on the player's position and risk.
{
    float score = 0.0;

    // Heatmap and space analysis
    pair<pair<vector<float>, vector<float>>, Matrix<float>> analysis = analyzer.analyzeState(playerState, currentPlayer);

    // Reachable space, blocking risk and heat risk
    float playerSpace = analysis.first.first[currentPlayer];
    float playerBlockRisk = analysis.first.second[currentPlayer];
    float playerHeatRisk = analyzer.calculateDirectionalRisk(analysis.second, playerState.at(currentPlayer).snake.front());


    if (currentPlayer == config.P)
    {
        // Positive reward if gain of space or reduce blocking risk and negative if lose space and increase blocking risk
        score += playerSpace * params.rewards.selfSpace;
        score += playerBlockRisk * params.rewards.selfBlockRisk;
        score += playerHeatRisk * params.rewards.selfHeatRisk;

        // reward the player for surviving (the longer the player survives the bigger the reward)
        score += playerState.at(config.P).snake.size() * params.rewards.snakeLength;
    }
    else
    {
        score -= playerSpace * params.rewards.oppSpace;
        score -= playerBlockRisk * params.rewards.oppBlockRisk;
        score -= playerHeatRisk * params.rewards.oppHeatRisk;
    }

    return score;
}  // ----- End of evaluateState


float Minimax::minimax(const vector<float> &boardState, Players playerState,
                         int depth, float alpha, float beta, int currentPlayer, int turn)
// Algorithm : Recursive miniMax with alpha-beta prunning to reduce the search space
{
    // check if the depth is 0 or a terminal state is reached
    if (depth == 0 || gameEngine.terminalState()) {
        return evaluateState(playerState, currentPlayer);
    }

    bool isMaximizing = (currentPlayer == config.P);  // check if the player of the node is the maximizing player
    float bestValue = (isMaximizing) ? -numeric_limits<float>::infinity() : numeric_limits<float>::infinity();

    // Get valid moves for the current player
    vector<string> moves = helper.getValidMoves(boardState, playerState, currentPlayer);  // todo: modify

    for (const auto &move : moves) {
        // Simulate the move
        auto simulatedState = simulateMove(move, boardState, playerState, turn, currentPlayer);
        int nextPlayer = helper.getNextPlayer(currentPlayer, simulatedState.second);  // todo: modify

        float eval;
        if (nextPlayer == -1) {
            // if a terminal state is reached, evaluate the state
            eval = evaluateState(simulatedState.second, currentPlayer);
        } else {
            // else, run the minimax algorithm
            eval = minimax(simulatedState.first, simulatedState.second,
                            depth - 1, alpha, beta, nextPlayer, turn + 1);
        }

        // Update the best value, alpha and beta based on the player
        if (isMaximizing) {
            bestValue = max(bestValue, eval);
            alpha = max(alpha, bestValue);
        } else {
            bestValue = min(bestValue, eval);
            beta = min(beta, bestValue);
        }

        if (beta <= alpha) break; // Prunning
    }

    return bestValue;
}  // ----- End of alphabeta