/*************************************************************************
Observation - Manages the board representation of the game
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//-- Implementation of the module <Observation> (file Observation.cpp) --

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <iostream>

using namespace std;

//------------------------------------------------------- Personal Include
#include "Observation.h"

//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
Matrix<vector<float>> Observation::getObservation() const
// Algorithm : Compute the board state
// board state is a 3D array with 9 channels per cell:
// 0: my snake head (0 or 1),
// 1: my snake body (0 or 1),
// 2: my snake tail (0 or 1),
// 3: opponent head (0 or 1),  // todo: value between 0.01 and 1.0 for the opponent's turn
// 4: opponent body (0 or 1),  // todo: same
// 5: opponent tail (0 or 1),  // todo: same
// 6: cell is accessible (0 or 1),  // todo
// 7: x coordinate of my snake head (normalized by the board width - 1),
// 8: y coordinate of my snake head (normalized by the board height - 1),
{
    return board;
}  // ----- End of getObservation

Matrix<vector<float>> Observation::initBoard(const std::vector<Coord> &playersOrigins)
// Algorithm : Initialize the board with the players' snakes
{
    vector<float> channelsTemplate(boardChannels, 0.0);

    pair<int, int> accessiblesCellInds = getAccessibleCellInd();
    for (int accessiblesCellInd = accessiblesCellInds.first; accessiblesCellInd <= accessiblesCellInds.second; accessiblesCellInd++)
    {
        channelsTemplate[accessiblesCellInd] = 1.0;  // Mark the cell as accessible
    }

    // Set my snake head coordinates (normalized)
    Coord myHead = playersOrigins[config.P];
    float xNorm = config.W <= 1 ? 0.0 : myHead.x / float(config.W-1);
    float yNorm = config.H <= 1 ? 0.0 : myHead.y / float(config.H-1);

    pair<int, int> myHeadXInds = getMyHeadXInd();
    pair<int, int> myHeadYInds = getMyHeadYInd();
    printf("> myHead: (%d, %d), xNorm: %.2f, yNorm: %.2f\n", myHead.x, myHead.y, xNorm, yNorm);
    for (int myHeadXInd = myHeadXInds.first; myHeadXInd <= myHeadXInds.second; myHeadXInd++)
    {
        channelsTemplate[myHeadXInd] = xNorm;
    }
    for (int myHeadYInd = myHeadYInds.first; myHeadYInd <= myHeadYInds.second; myHeadYInd++)
    {
        channelsTemplate[myHeadYInd] = yNorm;
    }

    board = Matrix(
        config.W, config.H, channelsTemplate
    );

    for (int i = 1; i <= config.N; ++i)
    {
        int x = playersOrigins[i].x;
        int y = playersOrigins[i].y;

        // Set the player's origin on the board
        pair<int, int> headInds = getPlayerHeadInd(i);
        printf("> Setting player %d head at (%d, %d) on channels [%d, %d]\n", i, x, y, headInds.first, headInds.second);
        for (int headInd = headInds.first; headInd <= headInds.second; headInd++)
        {
            board[x][y][headInd] = 1.0;  // Set the player's head
        }

        // Set the player's head as not accessible
        for (int accessiblesCellInd = accessiblesCellInds.first; accessiblesCellInd <= accessiblesCellInds.second; accessiblesCellInd++)
        {
            board[x][y][accessiblesCellInd] = 0.0;  // Mark the cell as not accessible
        }
    }

    return board;
}  // ----- End of initBoard

Matrix<vector<float>> Observation::updateBoard(const Players &playerState,
const int &currentPlayer, const Coord &prevHead, const int round)  // todo: modify to update observation 7, 8 and 9
// Algorithm : Update the snake's position based on the chosen action
{
    Coord newHead = playerState.at(currentPlayer).snake.front();

    // --- Update board ---
    pair<int, int> bodyInds = getPlayerBodyInd(currentPlayer);
    pair<int, int> headInds = getPlayerHeadInd(currentPlayer);
    // Replace previous head with body
    for (int bodyInd = bodyInds.first; bodyInd <= bodyInds.second; bodyInd++)
    {
        board[prevHead.x][prevHead.y][bodyInd] = 1.0;
    }
    for (int headInd = headInds.first; headInd <= headInds.second; headInd++)
    {
        board[prevHead.x][prevHead.y][headInd] = 0.0;

        // Update the new head position on the board
        board[newHead.x][newHead.y][headInd] = 1.0;
    }

    // --- Remove tail segments if it's not a growth round ---
    if (round % config.M != 0)
    {
        Coord prevTail = playerState.at(currentPlayer).snake.back();  // todo: problem : same as bellow

        // --- Update board ---
        pair<int, int> tailInds = getPlayerTailInd(currentPlayer);
        // Remove the tail segment from the board and modify the last body segment to ba a tail segment
        Coord newTail = playerState.at(currentPlayer).snake.back();
        for (int tailInd = tailInds.first; tailInd <= tailInds.second; tailInd++)
        {
            board[prevTail.x][prevTail.y][tailInd] = 0.0;

            board[newTail.x][newTail.y][tailInd] = 1.0;
        }
    }

    return board;
}  // ----- End of updateBoard
