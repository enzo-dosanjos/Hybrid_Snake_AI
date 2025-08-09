/*************************************************************************
GameEngine - Handles the game logic and operations
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//--- Implementation of the module <GameEngine> (file GameEngine.cpp) ---

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <iostream>

using namespace std;

//------------------------------------------------------- Personal Include
#include "../include/GameEngine.h"

//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
Matrix<vector<float>> GameEngine::initBoard(const std::vector<Coord> &playersOrigins)
// Algorithm : Initialize the board with the players' snakes
{
    board = Matrix<vector<float>>(
        config.W, config.H,
        vector<float>(boardChannels, 0.0)
    );

    for (int i = 1; i <= config.N; ++i)
    {
        int x = playersOrigins[i].x;
        int y = playersOrigins[i].y;

        // Set the player's origin on the board
        int headInd = observation.getPlayerHeadInd(i);
        board[x][y][headInd] = 1.0;  // Set the player's head
    }

    return board;
}

Players GameEngine::initPlayers(const vector<Coord> &playersOrigins)
// Algorithm : Initialize players' snakes and info
{
    players = Players(config.N);

    for (int i = 1; i <= config.N; ++i)
    {
        Coord playerOrigin = playersOrigins[i];
        players[i].snake.emplace_front(playerOrigin);

        // Initialise Player's data
        players[i].playerInfo.origin = playerOrigin;
        players[i].playerInfo.alive = true;
    }

    return players;
}

Players GameEngine::snakeUpdate(const int &currentPlayer, const string &move)  // modify to update observation 7, 8 and 9
// Algorithm : Update the snake's position based on the chosen action
{
    // --- Update players ---
    Coord prevHead = players[currentPlayer].snake.front();
    players[currentPlayer].snake.emplace_front(Coord{
        prevHead.x + DIRECTIONS.at(move).x,
        prevHead.y + DIRECTIONS.at(move).y
    });

    // --- Update board ---
    int bodyInd = observation.getPlayerBodyInd(currentPlayer);
    int headInd = observation.getPlayerHeadInd(currentPlayer);
    // Replace previous head with body
    board[prevHead.x][prevHead.y][bodyInd] = 1.0;
    board[prevHead.x][prevHead.y][headInd] = 0.0;

    // Update the new head position on the board
    Coord newHead = players[currentPlayer].snake.front();
    board[newHead.x][newHead.y][headInd] = 1.0;

    // --- Remove tail segments if it's not a growth round ---
    if (round % config.M != 0)
    {
        Coord prevTail = players[currentPlayer].snake.back();

        // --- Update players ---
        players[currentPlayer].snake.pop_back();

        // --- Update board ---
        int tailInd = observation.getPlayerTailInd(currentPlayer);
        // Remove the tail segment from the board
        board[prevTail.x][prevTail.y][tailInd] = 0.0;

        // Modify the last body segment to ba a tail segment
        Coord newTail = players[currentPlayer].snake.back();
        board[newTail.x][newTail.y][tailInd] = 1.0;

    }

    return players;
}  // ----- End of snakeUpdate

bool GameEngine::isInbound(int x, int y) const
// Algorithm : Check if the coordinates are in the board's bounds
{
    return x >= 0 && x < config.W && y >= 0 && y < config.H;
}  //----- End of isInbound


bool GameEngine::isDead(int playerNum) const
// Algorithm : Check if the player is dead by checking if the head of the snake
// is inbounds and if the cell is occupied by only one presence (i.e. my head is
// not on a cell occupied by another part of my body or the opponent's head)
{
    Coord playerHead = players.at(playerNum).snake.front();

    // check if the move is inbounds
    if (isInbound(playerHead.x, playerHead.y))
    {
        // Check if the cell is empty (no part of my body or no other player)
        pair<int, int> presenceInd = observation.getPresenceObservationsInd();
        vector<float> cell = board[playerHead.x][playerHead.y];

        int numPresence = 0;
        for (int i = presenceInd.first; i < presenceInd.second; i++) {
            if (cell[i] != 0.0)
            {
                numPresence++;
            }
        }

        if (numPresence > 1)  // if there is more than one presence in the cell
        {
            return false;
        }
    }

    return true;
}  //----- End of checkDeath

Matrix<vector<float>> GameEngine::Observation::getObservation(const Players &players)  // todo: modify to only return board
// Algorithm : Compute the board state
// board state is a flattened vector<float> with 10 channels per cell:
// 0: my snake head (0 or 1),
// 1: my snake body (0 or 1),
// 2: my snake tail (0 or 1),
// 3: opponent head (0 or 1),
// 4: opponent body (0 or 1),
// 5: opponent tail (0 or 1),
// 6: cell is accessible (0 or 1),  // todo
// 7: x coordinate of my snake head (normalized by the board width - 1),  // todo
// 8: y coordinate of my snake head (normalized by the board height - 1),  // todo
{
    Matrix<vector<float>> board(
        config.W, config.H,
        vector<float>(boardChannels, 0.0)
    );

    for (const auto &player : players)
    {
        const int id = player.first;
        const deque<Coord> &snake = player.second.snake;

        for (const auto &part : snake)
        {
            if (id == config.P)
            {
                if (part == snake.front())
                {
                    board[part.x][part.y][0] = 1.0;  // my snake head
                }
                else if (part == snake.back())
                {
                    board[part.x][part.y][2] = 1.0;  // my snake tail
                }
                else
                {
                    board[part.x][part.y][1] = 1.0;  // my snake body
                }
            }
            else
            {
                if (part == snake.front())
                {
                    board[part.x][part.y][3] = 1.0;  // opponent head
                }
                else if (part == snake.back())
                {
                    board[part.x][part.y][5] = 1.0;  // opponent tail
                }
                else
                {
                    board[part.x][part.y][4] = 1.0;  // opponent body
                }
            }
        }


    }

    return board;
}  // ----- End of computeBoardState

bool GameEngine::terminalState()
// Algorithm : Check if the game has reached a terminal state
{
    // Check if only one player is alive or if the current player is dead
    return (numPlayerAlive <= 1 || !players[config.P].playerInfo.alive);
}  // ----- End of terminalState

void GameEngine::PlayerSelector::nextPlayer()
{
    currentPlayer = (currentPlayer % config.N) + 1;

    turn++;

    // If we reached the last player, increment the round
    if (currentPlayer == config.N)
    {
        round++;
    }
}