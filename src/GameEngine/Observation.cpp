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

    // Each player retains head/body/tail channels and an ordered-segment channel.
    // The final channel marks free cells; head-coordinate broadcasts are unnecessary.
    int freeChannel = CHANNELS_PER_PLAYER * config.N;
    BoardTensor board(freeChannel + 1, Matrix<float>(config.W, config.H, 0.f));
    board[freeChannel] = Matrix<float>(config.W, config.H, 1.f);
    vector<int> players{perspective};

    for (int player = 1; player <= config.N; player++)
    {
        if (player != perspective)
        {
            players.push_back(player);
        }
    }

    for (size_t slot = 0; slot < players.size(); slot++)
    {
        const auto &player = state.players.at(players[slot]);
        int offset = CHANNELS_PER_PLAYER * slot;

        for (size_t segment = 0; segment < player.snake.size(); segment++)
        {
            Coord cell = player.snake[segment];

            if (!isInbound(cell, config))
            {
                throw invalid_argument("Snake segment outside the board");
            }

            // A single segment belongs to both the head and tail channels.
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

            // Zero means absent; multiplying by board area recovers the one-based segment index.
            board[offset + 3][cell.y][cell.x] = (segment + 1) / (float(config.W) * config.H);
            board[freeChannel][cell.y][cell.x] = 0.f;
        }
    }

    return board;
}  //----- end of encode
