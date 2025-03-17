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
#include <cmath>
#include <algorithm>

using namespace std;

//------------------------------------------------------- Personal Include
#include "../include/Minimax.h"

//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
unordered_map<string, float> Minimax::calculateMoveScores(vector<float> &boardState,
                                                          PlayersData &playerState, int turn)
// Algorithm : Main method to calculate the move scores and return the best move.
// It uses the minimax algorithm with alpha-beta prunning to reduce the search space.
{
    unordered_map<string, float> moveScores;

    // Get valid moves for the current player
    vector<string> validMoves = helper.getValidMoves(boardState, playerState, P);

    // Initialize alpha and beta values
    float alpha = -numeric_limits<float>::infinity();
    float beta = numeric_limits<float>::infinity();
    float bestValue = -numeric_limits<float>::infinity();

    for (const auto &move : validMoves) {
        // Simulate the move
        auto simulatedState = simulateMove(move, boardState, playerState, turn, P);
        int nextPlayer = helper.getNextPlayer(P, simulatedState.second);

        float eval;
        if (nextPlayer == -1)
        {
            // if a terminal state is reached, evaluate the state
            eval = evaluateState(simulatedState.second, P);
        }
        else
        {
            // else, run the minimax algorithm
            eval = alphabeta(simulatedState.first, simulatedState.second, params.depth - 1, alpha, beta, nextPlayer, turn + 1);
        }

        // Update the best value, alpha and beta
        bestValue = max(bestValue, eval);
        alpha = max(alpha, bestValue);

        if (beta <= alpha) break; // Prunning

        moveScores[move] = eval;
    }

    return moveScores;
}  // ----- End of calculateMoveScores


void Minimax::updateRewards(const MinimaxParams &newParams)
// Algorithm : Setter method to update the rewards and depth based on the new MinimaxParams parameters
{
    params = newParams;
}  // ----- End of updateRewards


void Minimax::updateRewards(float selfSpace, float oppSpace, float selfBlockRisk,
                            float oppBlockRisk, float selfHeatRisk, float oppHeatRisk, float snakeLength, int depth)
// Algorithm : Setter method to update the rewards and depth based on the new parameters values
{
    params.rewards.selfSpace = selfSpace;
    params.rewards.oppSpace = oppSpace;
    params.rewards.selfBlockRisk = selfBlockRisk;
    params.rewards.oppBlockRisk = oppBlockRisk;
    params.rewards.selfHeatRisk = selfHeatRisk;
    params.rewards.oppHeatRisk = oppHeatRisk;
    params.rewards.snakeLength = snakeLength;
    params.depth = depth;
}  // ----- End of updateRewards


MinimaxParams Minimax::findClosestConfig(const string &filename)
// Algorithm: Find the closest board configuration [W,H,M]
{
    MinimaxParams closestParams;
    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "> Error: parameters file not found!\n";
        return closestParams;
    }

    string line;
    double minDist= numeric_limits<double>::max();
    string closestConfig;
    streampos configPos = 0;

    // Find all configurations and calculate distance
    while (getline(file, line))
    {
        if (!line.empty() && line[0] == '[')
        {
            // Store position before reading this line
            streampos currentPos = file.tellg();
            currentPos -= (line.length() + 1); // Account for the line we just read

        	// Parse [W,H,M] from line
        	size_t start = line.find('[') + 1;
        	size_t end = line.find(']');

        	if (start != string::npos && end != string::npos)
            {
                string configStr = line.substr(start, end - start);
        		stringstream ss(configStr);

    	    	int w, h, m;
    	    	char comma;
    	    	ss >> w >> comma >> h >> comma >> m;

    	    	// Calculate normalized Euclidean distance
    	    	double distance = pow((W - w)/50.0, 2) +
    	    	    			  pow((H - h)/50.0, 2) +
    	    	    			  pow((M - m)/5.0, 2);

    	    	if (distance < minDist)
                {
        		    minDist = distance;
        		    closestConfig = line;
                    configPos = currentPos;
        		}
            }
        }
    }

    // Return the parameters for the closest configuration
    if (!closestConfig.empty())
    {
        closestParams = loadParameters(closestConfig, file, configPos);
    }

    return closestParams;
}  // ----- End of find_closest_config


//-------------------------------------------------------------- PROTECTED
MinimaxParams Minimax::loadParameters(const string &config, ifstream &file, streampos filePos)
// Algorithm: Load parameters for the specified configuration based on player number
{
    MinimaxParams newParams;
    string currentLine;
    bool foundConfig = false;

    // Clear any error flags and set file position
    file.clear();

    // Find the configuration block
    if (filePos != 0) {
        // If a position was provided, use it
        file.seekg(filePos);
        foundConfig = true;
    } else {
        // Otherwise search for the config from the beginning
        file.seekg(0, std::ios::beg);

        while (getline(file, currentLine))
        {
            if (currentLine == config) {
                foundConfig = true;
                break;
            }
        }
    }

    if (!foundConfig)
    {
        cout << "> Config " << config << " not found!\n";
        return newParams;
    }

    // Read AI parameters
    string aiPrefix = (P == 1) ? "AI1" : "AI2";
    string depthPrefix = aiPrefix + "Depth=";

    while (getline(file, currentLine))
    {
        if (!currentLine.empty() && currentLine[0] != '[')
        {
            // Parse parameters
        	if (currentLine.find(aiPrefix + "{") != string::npos)
            {
        	    parseParameters(currentLine, newParams);
        	}
        	// Parse depth
        	else if (currentLine.find(depthPrefix) != string::npos)
                {
        	    size_t pos = currentLine.find('=');
        	    if (pos != string::npos)
                {
        	        try
                    {
        	            newParams.depth = stoi(currentLine.substr(pos + 1));
        	        }
                    catch (...)
                    {
        	            cout << "> Invalid depth format, using default depth=8\n";
        	            newParams.depth = 8;
        	        }
        	    }
        	}
        }
        else if (!currentLine.empty() && currentLine[0] == '[')
        {
            // reached the next configuration block
            break;
        }
    }

    return newParams;
}  // ----- End of load_parameters


void Minimax::parseParameters(const string& line, MinimaxParams& newParams)
// Algorithm: Parse reward parameters from an AI{param1,param2,...} line
{
    size_t start = line.find('{') + 1;
    size_t end = line.find('}');

    if (start == string::npos || end == string::npos || start >= end)
    {
        cout << "> Invalid parameter format: " << line << endl;
        return;
    }

    string paramStr = line.substr(start, end - start);
    replace(paramStr.begin(), paramStr.end(), ',', ' ');

    stringstream ss(paramStr);
    ss >> newParams.rewards.selfSpace
       >> newParams.rewards.oppSpace
       >> newParams.rewards.selfBlockRisk
       >> newParams.rewards.oppBlockRisk
       >> newParams.rewards.selfHeatRisk
       >> newParams.rewards.oppHeatRisk
       >> newParams.rewards.snakeLength;
}  // ----- End of parse_parameters


pair<vector<float>, PlayersData> Minimax::simulateMove(const string &move, const vector<float> &boardState,
                                                       const PlayersData &playerState, int turn, int playerNb)
// Algorithm : Simulate the move of a player and update the board and player states.
{
    // Make a copy of the current state
    pair<vector<float>, PlayersData> newState;
    newState.first = boardState;
    newState.second = playerState;


    // Get current head and tail positions
    Coord currentHead = newState.second[playerNb].first.front();
    Coord currentTail = newState.second[playerNb].first.back();

    // create new head position from the move
    Coord newHead = make_pair(currentHead.first + DIRECTIONS.at(move).first,
                              currentHead.second + DIRECTIONS.at(move).second
                             );


    // Update snake based on chosen move
    newState.second[playerNb].first.push_front(newHead);

    // Count alive players after the move
    int alivePlayers = 0;
    for (const auto &player : newState.second)
    {
        if (!helper.checkDeath(newState.second, newState.first, player.first))
        {
            alivePlayers++;
        }
        else
        {
            newState.second[player.first].second.alive = false;
        }
    }

    int currentRound = ((turn) / alivePlayers); // turn+1 because this move increments the turn


    // Remove tail segment if it's not a growth round from player states
    if (currentRound % M != 0)
    {
        newState.second[playerNb].first.pop_back();
    }


    int ind = (currentHead.second * W + currentHead.first) * nbBoardChannels;  // current head
    int newInd = (newHead.second * W + newHead.first) * nbBoardChannels;  // new head

    // Make the current head become a body part and create the new head
    if (playerNb == P)
    {
        if (newState.first[ind] == 1.0)
        {
            newState.first[ind] = 0.0;
            newState.first[ind + 1] = 1.0;
        }
        newState.first[newInd] = 1.0;

        // remove tail segment if it's not a growth round from board state
        if (currentRound % M != 0)
        {
            newState.first[(currentTail.second * W + currentTail.first) * nbBoardChannels + 1] = 0.0;
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

        if (currentRound % M != 0)
        {
            newState.first[(currentTail.second * W + currentTail.first) * nbBoardChannels + 3] = 0.0;
        }
    }

    return newState;
}  // ----- End of simulateMove


float Minimax::evaluateState(const PlayersData &playerState, int currentPlayer)
// Algorithm : Evaluate the state of the game and return a score based on the player's position and risk.
{
    float score = 0.0;

    // Heatmap and space analysis
    pair<pair<vector<float>, vector<float>>, Matrix<float>> analysis = analyzer.analyzeState(playerState, currentPlayer);

    // Reachable space, blocking risk and heat risk
    float playerSpace = analysis.first.first[currentPlayer];
    float playerBlockRisk = analysis.first.second[currentPlayer];
    float playerHeatRisk = analyzer.calculateDirectionalRisk(analysis.second, playerState.at(currentPlayer).first.front());


    if (currentPlayer == P)
    {
        // Positive reward if gain of space or reduce blocking risk and negative if lose space and increase blocking risk
        score += playerSpace * params.rewards.selfSpace;
        score += playerBlockRisk * params.rewards.selfBlockRisk;
        score += playerHeatRisk * params.rewards.selfHeatRisk;

        // reward the player for surviving (the longer the player survives the bigger the reward)
        score += playerState.at(P).first.size() * params.rewards.snakeLength;
    }
    else
    {
        score -= playerSpace * params.rewards.oppSpace;
        score -= playerBlockRisk * params.rewards.oppBlockRisk;
        score -= playerHeatRisk * params.rewards.oppHeatRisk;
    }

    return score;
}  // ----- End of evaluateState


float Minimax::alphabeta(const vector<float> &boardState, PlayersData playerState,
                         int depth, float alpha, float beta, int currentPlayer, int turn)
// Algorithm : Recursive miniMax with alpha-beta prunning to reduce the search space
{
    // check if the depth is 0 or a terminal state is reached
    if (depth == 0 || helper.checkTerminalState(boardState, playerState)) {
        return evaluateState(playerState, currentPlayer);
    }

    bool isMaximizing = (currentPlayer == P);  // check if the player of the node is the maximizing player
    float bestValue = (isMaximizing) ? -numeric_limits<float>::infinity() : numeric_limits<float>::infinity();

    // Get valid moves for the current player
    vector<string> moves = helper.getValidMoves(boardState, playerState, currentPlayer);

    for (const auto &move : moves) {
        // Simulate the move
        auto simulatedState = simulateMove(move, boardState, playerState, turn, currentPlayer);
        int nextPlayer = helper.getNextPlayer(currentPlayer, simulatedState.second);

        float eval;
        if (nextPlayer == -1) {
            // if a terminal state is reached, evaluate the state
            eval = evaluateState(simulatedState.second, currentPlayer);
        } else {
            // else, run the minimax algorithm
            eval = alphabeta(simulatedState.first, simulatedState.second,
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