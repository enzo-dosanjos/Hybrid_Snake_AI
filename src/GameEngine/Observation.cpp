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
        pair<int, int> tailInds = getPlayerTailInd(i);

        for (int headInd = headInds.first; headInd <= headInds.second; headInd++)
        {
            board[x][y][headInd] = 1.0;  // Set the player's head
        }

        for (int tailInd = tailInds.first; tailInd <= tailInds.second; tailInd++)
        {
            board[x][y][tailInd] = 1.0;  // Set the player's tail
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
const int &currentPlayer, const int &round)
// Algorithm : Update the snake's position based on the chosen action
{
    // if the player is dead, do nothing
    if (!playerState.at(currentPlayer).playerInfo.alive) {
        return board;
    }

    Coord newHead = playerState.at(currentPlayer).snake.front();
    Coord newTail = playerState.at(currentPlayer).snake.back();

    // Find the previous head and tail positions using the stored board state
    Coord prevHead = findPrevSegment(newHead, getPlayerHeadInd(currentPlayer));
    Coord prevTail = findPrevSegment(newTail, getPlayerTailInd(currentPlayer));


    // --- Update board ---
    pair<int, int> bodyInds = getPlayerBodyInd(currentPlayer);
    pair<int, int> headInds = getPlayerHeadInd(currentPlayer);
    pair<int, int> accessiblesCellInds = getAccessibleCellInd();
    // Replace previous head with body
    for (int bodyInd = bodyInds.first; bodyInd <= bodyInds.second; bodyInd++)
    {
        if (playerState.at(currentPlayer).snake.size() > 2)
        {
            board[prevHead.x][prevHead.y][bodyInd] = 1.0;
        }

    }
    for (int headInd = headInds.first; headInd <= headInds.second; headInd++)
    {
        board[prevHead.x][prevHead.y][headInd] = 0.0;

        // Update the new head position on the board
        board[newHead.x][newHead.y][headInd] = 1.0;
    }

    // Mark new head cell as not accessible
    for (int accessiblesCellInd = accessiblesCellInds.first; accessiblesCellInd <= accessiblesCellInds.second; accessiblesCellInd++)
    {
        board[newHead.x][newHead.y][accessiblesCellInd] = 0.0;
    }

    // --- Remove tail segments if it's not a growth round ---
    if (round % config.M != 0)
    {

        // --- Update board ---
        pair<int, int> tailInds = getPlayerTailInd(currentPlayer);
        // Remove the tail segment from the board and modify the last body segment to ba a tail segment
        for (int tailInd = tailInds.first; tailInd <= tailInds.second; tailInd++)
        {
            board[prevTail.x][prevTail.y][tailInd] = 0.0;

            board[newTail.x][newTail.y][tailInd] = 1.0;
        }

        // Update the previous body cell to be a tail cell
        for (int bodyInd = bodyInds.first; bodyInd <= bodyInds.second; bodyInd++)
        {
            board[newTail.x][newTail.y][bodyInd] = 0.0;
        }

        // Update the previous tail cell to be accessible
        for (int accessiblesCellInd = accessiblesCellInds.first; accessiblesCellInd <= accessiblesCellInds.second; accessiblesCellInd++)
        {
            board[prevTail.x][prevTail.y][accessiblesCellInd] = 1.0;
        }
    }

    return board;
}  // ----- End of updateBoard

Coord Observation::findPrevSegment(const Coord& segment, const pair<int,int>& inds) const {
    for (int move = 0; move < static_cast<int>(ACTIONS.size()); ++move) {
        Coord neighbor = computeNewCoord(segment, move);
        if (!isInbound(neighbor, config)) {
            continue;
        }
        bool matches = true;
        for (int idx = inds.first; idx <= inds.second; ++idx) {
            if (board[neighbor.x][neighbor.y][idx] == 0.0f) {
                matches = false;
                break;
            }
        }

        if (matches) {
            return neighbor;
        }
    }

    return segment;
}
