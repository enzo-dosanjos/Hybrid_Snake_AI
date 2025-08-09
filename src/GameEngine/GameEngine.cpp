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
#include "GameEngine.h"

//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
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
        players[i].playerInfo.actionMask = vector<bool>(ACTIONS.size(), true);  // All actions are valid at the start
    }

    return players;
}

Players GameEngine::updateSnake(const int &currentPlayer, const Coord &newHead)
// Algorithm : Update the snake's position based on the chosen action
{
    // --- Update players ---
    players[currentPlayer].snake.emplace_front(newHead);

    // --- Remove tail segments if it's not a growth round ---
    if (round % config.M != 0)
    {
        // --- Update players ---
        players[currentPlayer].snake.pop_back();
    }

    return players;
}  // ----- End of updateSnake

vector<bool> GameEngine::updateActionMask(const int &currentPlayer, const Coord &newHead)
// Algorithm : Update the action mask for the current player
// The action mask indicates which actions are valid for the current player
{
    vector<bool> &actionMask = players[currentPlayer].playerInfo.actionMask;

    for (int i = 0; i < ACTIONS.size(); i++)
    {
        // todo: compute new head for each action and check if it's valid
        actionMask[i] = isAlive(currentPlayer);
    }

    return actionMask;
}  // ----- End of updateActionMask

Players GameEngine::updateStep(const int &currentPlayer, const int &move)
// Algorithm : Update the game state. This method is called after each step
{
    // Compute the new head position based on the current move
    Coord currentHead = players[currentPlayer].snake.front();
    Coord newHead = computeNewCoord(currentHead, move);

    // Update players
    updateSnake(currentPlayer, newHead);
    updateActionMask(currentPlayer, newHead);
    players[currentPlayer].playerInfo.alive = isAlive(currentPlayer);

    // Update the round and turn counters
    turn++;

    // If we reached the last player, increment the round and reset the turn
    if (turn == config.N - 1)
    {
        round++;
        turn = 0;
    }

    return players;
}  // ----- End of updateStep

Coord GameEngine::computeNewCoord(const Coord &currentHead, const int &move)
// Algorithm : Computes the new coordinates of the snake's head based on the current head and the
// move
{
    Coord newHead = {
        currentHead.x + DIRECTIONS[move].x,
        currentHead.y + DIRECTIONS[move].y
    };

    return newHead;
}  //----- End of computeNewCoord

bool GameEngine::isInbound(Coord coord) const
// Algorithm : Check if the coordinates are in the board's bounds
{
    return coord.x >= 0 && coord.x < config.W &&
        coord.y >= 0 && coord.y < config.H;
}  //----- End of isInbound


bool GameEngine::isAlive(const int &playerNum, const Coord &head) const
// Algorithm : Check if the player is alive by checking if the head of the snake
// is inbounds and if the cell is occupied by only one presence (i.e. my head is
// not on a cell occupied by another part of my body or the opponent's head)
{
    Coord currentHead;
    if (head.x == -1 && head.y == -1)
    {
        Coord currentHead = players.at(playerNum).snake.front();
    }
    else
    {
        currentHead = head;
    }

    // check if the player is alive and the move is inbounds
    if (players.at(playerNum).playerInfo.alive && isInbound(currentHead))
    {
        // Check if the head collides with any snake body part
        for (const auto &pair : players)
        {
            int otherPlayerNum = pair.first;
            const PlayersData &otherPlayerData = pair.second;

            // Check each body part of the other player
            for (const Coord &bodyPart : otherPlayerData.snake)
            {
                // Skip checking the head against itself
                if (otherPlayerNum == playerNum && bodyPart == currentHead)
                {
                    continue;
                }

                if (bodyPart == currentHead)
                {
                    return false;  // Collision detected
                }
            }
        }
    }

    return false;
}  //----- End of isAlive

bool GameEngine::isTerminalState()
// Algorithm : Check if the game has reached a terminal state
{
    // Check if only one player is alive or if the current player is dead
    return (numPlayerAlive <= 1 || !players[config.P].playerInfo.alive);
}  // ----- End of isTerminalState
