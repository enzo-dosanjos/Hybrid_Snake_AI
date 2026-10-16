/*************************************************************************
Observation - Encodes a game snapshot as board channels
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//-- Implementation of the module <Observation> (file Observation.cpp) --

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <stdexcept>

using namespace std;

//------------------------------------------------------- Personal Include
#include "Observation.h"
#include "../Utils.h"

//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
BoardTensor Observation::encode(const GameState &state, int perspective) const
// Algorithm : Rebuild segment, empty-cell and head-coordinate channels directly from the supplied snapshot.
{
    validateGameConfig(state.config);
    const auto &config = state.config;
    const auto &snake = state.players.at(perspective).snake;

    if (snake.empty())
    {
        throw invalid_argument("Cannot encode an empty snake");
    }

    BoardTensor board(BOARD_CHANNELS, Matrix<float>(config.W, config.H, 0.f));
    Coord head = snake.front();
    float headX = config.W == 1 ? 0.f : head.x / float(config.W - 1);
    float headY = config.H == 1 ? 0.f : head.y / float(config.H - 1);

    for (int y = 0; y < config.H; y++)
    {
        for (int x = 0; x < config.W; x++)
        {
            board[6][y][x] = 1.f;
            board[7][y][x] = headX;
            board[8][y][x] = headY;
        }
    }

    for (const auto &[id, player] : state.players)
    {
        int offset = id == perspective ? 0 : 3;

        for (size_t segment = 0; segment < player.snake.size(); segment++)
        {
            Coord cell = player.snake[segment];

            if (!isInbound(cell, config))
            {
                throw invalid_argument("Snake segment outside the board");
            }

            // A single segment belongs to both channels; opponent channels merge all other players.
            if (segment == 0)
            {
                board[offset][cell.y][cell.x] = 1.f;
            }

            if (segment + 1 == player.snake.size())
            {
                board[offset + 2][cell.y][cell.x] = 1.f;
            }

            if (segment > 0 && segment + 1 < player.snake.size())
            {
                board[offset + 1][cell.y][cell.x] = 1.f;
            }

            board[6][cell.y][cell.x] = 0.f;
        }
    }

    return board;
}  //----- end of encode
