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

    // Now that all players are initialized, set the initial action masks
    for (int i = 1; i <= config.N; ++i)
    {
        Coord head = players[i].snake.front();
        updateActionMask(i, head);
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

vector<bool> GameEngine::updateActionMask(const int &currentPlayer, const Coord &head)
// Algorithm : Update the action mask for the current player
// The action mask indicates which actions are valid for the current player
{
    vector<bool> &actionMask = players[currentPlayer].playerInfo.actionMask;

    for (size_t i = 0; i < ACTIONS.size(); i++)
    {
        Coord potentialNewHead = computeNewCoord(head, i);
        actionMask[i] = isAlive(currentPlayer, potentialNewHead);
    }

    return actionMask;
}  // ----- End of updateActionMask

Players GameEngine::updateStep(const int &currentPlayer, const int &move)
// Algorithm : Update the game state. This method is called after each step
{
    if (!players[currentPlayer].playerInfo.alive)
    {
        // If the current player is already dead, no need to update
        return players;
    }

    // Compute the new head position based on the current move
    Coord currentHead = players[currentPlayer].snake.front();
    Coord newHead = computeNewCoord(currentHead, move);

    // Update players
    updateSnake(currentPlayer, newHead);

    // Update action mask for all players
    for (int i = 1; i <= config.N; ++i)
    {
        Coord head = players[i].snake.front();
        updateActionMask(i, head);
    }

    if (currentPlayer == config.P)
    {
        // Check if the AI will die after this move
        if (willDieNextTurn(currentPlayer))
        {
            playerDied(currentPlayer);
        }
    }

    // Update the round and turn counters
    turn++;

    // If we reached the last player, increment the round and reset the turn
    if (turn == config.N)
    {
        round++;
        turn = 0;
    }

    return players;
}  // ----- End of updateStep

Players GameEngine::playerDied(const int &playerNum)
// Algorithm : Mark the player as dead and update the number of alive players
{
    if (players[playerNum].playerInfo.alive)
    {
        players[playerNum].playerInfo.alive = false;
        numPlayerAlive--;
    }

    return players;
}  // ----- End of playerDied

bool GameEngine::isAlive(const int &playerNum, const Coord &head) const
// Algorithm : Check if the player is alive by checking if the head of the snake
// is inbounds and if the cell is occupied by only one presence (i.e. my head is
// not on a cell occupied by another part of my body or the opponent's head)
{
    Coord currentHead;
    if (head.x == -1 && head.y == -1)
    {
        currentHead = players.at(playerNum).snake.front();
    }
    else
    {
        currentHead = head;
    }

    // check if the move is inbounds
    if (isInbound(currentHead, config))
    {
        // Check if the head collides with any snake body part
        for (const auto &otherPlayer : players)
        {
            int otherPlayerNum = otherPlayer.first;
            const PlayersData &otherPlayerData = otherPlayer.second;

            // Check each body part of the other player
            for (const Coord &bodyPart : otherPlayerData.snake)
            {
                // Skip checking the head against itself
                if (otherPlayerNum == playerNum && &bodyPart == &otherPlayerData.snake.front())
                {
                    continue;
                }

                if (bodyPart == currentHead)
                {
                    return false;  // Collision detected
                }
            }
        }

        return true;  // No collision detected, player is alive
    }

    return false;
}  //----- End of isAlive

bool GameEngine::willDieNextTurn(const int &playerNum) const
// Algorithm : Check if the player will die in the next turn using the action mask
{
    bool willDie = true;
    for (bool actionValid : players.at(playerNum).playerInfo.actionMask)
    {
        if (actionValid)
        {
            willDie = false;
            break;
        }
    }

    return willDie;
}  // ----- End of willDieNextTurn

bool GameEngine::isTerminalState()
// Algorithm : Check if the game has reached a terminal state
{
    // Check if only one player is alive or if the current player is dead
    return (numPlayerAlive <= 1 || !players[config.P].playerInfo.alive);
}  // ----- End of isTerminalState
