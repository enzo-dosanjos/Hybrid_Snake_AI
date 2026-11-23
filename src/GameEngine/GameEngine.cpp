/*************************************************************************
GameEngine - Applies movement, collision, growth and turn scheduling
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//--- Implementation of the module <GameEngine> (file GameEngine.cpp) ---

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <stdexcept>

using namespace std;

//------------------------------------------------------- Personal Include
#include "GameEngine.h"

//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
void GameEngine::updateStep(Action move)
// Algorithm : Check collision before movement, apply growth and history, then advance exactly one turn.
{
    int action = static_cast<int>(move);

    if (action < 0 || action >= static_cast<int>(ACTIONS.size()))
    {
        throw invalid_argument("Invalid action");
    }

    int player = state.currentPlayer;

    if (isTerminalFor(player))
    {
        throw logic_error("Cannot move in a terminal game or a dead player's slot");
    }

    if (!getActionMask(player)[action])
    {
        // The runner checks collisions before changing the snake or releasing its tail.
        playerDied(player);
    }
    else
    {
        Coord head = state.players.at(player).snake.front();
        updateSnake(player, computeNewCoord(head, move));
        auto &history = state.lastMoves.at(player);
        history.push_back(move);

        if (history.size() > ACTION_HISTORY_SIZE)
        {
            history.pop_front();
        }

        updateActionMasks();
    }

    finishTurn();
}  //----- end of updateStep


void GameEngine::playerDied(int playerNum)
// Algorithm : Clear the alive flag and legal moves while keeping the body and schedule unchanged.
{
    auto &info = state.players.at(playerNum).playerInfo;
    info.alive = false;
    info.actionMask.reset();
}  //----- end of playerDied


void GameEngine::finishTurn()
// Algorithm : Delegate one scheduled-slot advancement to the player selector.
{
    selector.advance(state);
}  //----- end of finishTurn


bool GameEngine::isAlive(int playerNum) const
// Algorithm : Read the authoritative alive flag for the requested player.
{
    return state.players.at(playerNum).playerInfo.alive;
}  //----- end of isAlive


bool GameEngine::willDieNextTurn(int playerNum) const
// Algorithm : Check whether the current position offers any legal move to the player.
{
    return getActionMask(playerNum).none();
}  //----- end of willDieNextTurn


bool GameEngine::isTerminalState() const
// Algorithm : End the game when at most one player remains alive.
{
    return getNumPlayerAlive() <= 1;
}  //----- end of isTerminalState


bool GameEngine::isTerminalFor(int playerNum) const
// Algorithm : End a player's episode on its death or on global game termination.
{
    return !isAlive(playerNum) || isTerminalState();
}  //----- end of isTerminalFor


GameEngine GameEngine::successor(Action move) const
// Algorithm : Copy the complete engine and apply the same transition used in real play.
{
    GameEngine next = *this;
    next.updateStep(move);
    return next;
}  //----- end of successor


int GameEngine::getNumPlayerAlive() const
// Algorithm : Count alive flags directly so the count cannot drift from player data.
{
    int count = 0;

    for (const auto &[id, player] : state.players)
    {
        if (player.playerInfo.alive)
        {
            count++;
        }
    }

    return count;
}  //----- end of getNumPlayerAlive


//---------------------------------------------------------------- PRIVATE
void GameEngine::initPlayers(const vector<Coord> &playersOrigins)
// Algorithm : Initialize one-based players at distinct free cells and create their empty action histories.
{
    if (playersOrigins.size() <= static_cast<size_t>(state.config.N))
    {
        throw invalid_argument("Missing player origins");
    }

    for (int player = 1; player <= state.config.N; player++)
    {
        Coord origin = playersOrigins[player];

        if (!isFree(origin))
        {
            throw invalid_argument("Player origins must be distinct and inside the board");
        }

        auto &data = state.players[player];
        data.snake.emplace_front(origin);
        data.playerInfo.origin = origin;
        data.playerInfo.alive = true;
        state.lastMoves[player] = {};
    }
}  //----- end of initPlayers


void GameEngine::updateSnake(int currentPlayer, const Coord &newHead)
// Algorithm : Prepend the new head and release the tail only on a non-growth round.
{
    auto &snake = state.players.at(currentPlayer).snake;
    snake.emplace_front(newHead);

    if (state.round % state.config.M != 0)
    {
        snake.pop_back();
    }
}  //----- end of updateSnake


void GameEngine::updateActionMasks()
// Algorithm : Recompute legal neighbors for living players using the current occupancy.
{
    for (auto &[id, player] : state.players)
    {
        player.playerInfo.actionMask.reset();

        if (!player.playerInfo.alive)
        {
            continue;
        }

        for (size_t action = 0; action < ACTIONS.size(); action++)
        {
            Coord cell = computeNewCoord(player.snake.front(), static_cast<Action>(action));
            player.playerInfo.actionMask[action] = isFree(cell);
        }
    }
}  //----- end of updateActionMasks


bool GameEngine::isFree(const Coord &cell) const
// Algorithm : Reject walls and every occupied segment, including the bodies of dead snakes.
{
    if (!isInbound(cell, state.config))
    {
        return false;
    }

    for (const auto &[id, player] : state.players)
    {
        for (const Coord &part : player.snake)
        {
            if (part == cell)
            {
                return false;
            }
        }
    }

    return true;
}  //----- end of isFree
