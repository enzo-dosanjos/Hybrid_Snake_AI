/*************************************************************************
GameEngine - Handles the game logic and operations. Owns the authoritative
             game state and simulation rules
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//------- Interface of the module <GameEngine> (file GameEngine.h) -------
#ifndef GAMEENGINE_H
#define GAMEENGINE_H

//------------------------------------------------------------------------
// Role of the <GameEngine> module
// Applies game rules to a complete copyable state shared by real play and
// search.
//------------------------------------------------------------------------

//-------------------------------------------------------- Used interfaces
#include "../Structs.h"
#include "../Utils.h"
#include "PlayerSelector.h"

//----------------------------------------------------------------- PUBLIC
class GameEngine
{
//--------------------------------------------------------- Public Methods
    public:
        void updateStep(Action move);
        // Contract : Apply the current living player's action and finish exactly one scheduled turn.

        void playerDied(int playerNum);
        // Contract : Mark the player dead without advancing the clock or removing its body.

        void finishTurn();
        // Contract : Advance one scheduled slot, including a dead player's slot.

        bool isAlive(int playerNum) const;
        // Contract : Return the stored alive status, independently of available moves.

        bool willDieNextTurn(int playerNum) const;
        // Contract : Report whether the player has no legal move in the current position.

        bool isTerminalState() const;
        // Contract : A game is terminal when at most one player survives.

        bool isTerminalFor(int playerNum) const;
        // Contract : A player's episode ends on death or when the game ends.

        GameEngine successor(Action move) const;
        // Contract : Apply the action to an independent copy without changing this engine.

//--------------------------------------------------------- SETTERS/GETTERS
        const GameState &getState() const
        // Algorithm : Expose the complete snapshot without allowing external mutation.
        {
            return state;
        }  //----- end of getState


        const GameConfig &getConfig() const
        // Algorithm : Return the configuration held by the authoritative state.
        {
            return state.config;
        }  //----- end of getConfig


        int getW() const
        // Algorithm : Return the board width.
        {
            return state.config.W;
        }  //----- end of getW


        int getH() const
        // Algorithm : Return the board height.
        {
            return state.config.H;
        }  //----- end of getH


        int getM() const
        // Algorithm : Return the number of rounds between growth moves.
        {
            return state.config.M;
        }  //----- end of getM


        int getN() const
        // Algorithm : Return the original number of scheduled players.
        {
            return state.config.N;
        }  //----- end of getN


        int getP() const
        // Algorithm : Return the controlled player's one-based ID.
        {
            return state.config.P;
        }  //----- end of getP


        int getRound() const
        // Algorithm : Return the round containing the next scheduled turn.
        {
            return state.round;
        }  //----- end of getRound


        int getTurn() const
        // Algorithm : Derive the zero-based slot within the current round.
        {
            return state.currentPlayer - 1;
        }  //----- end of getTurn


        int getNumPlayerAlive() const;
        // Contract : Count living players without maintaining a separate cached counter.

        const Players &getPlayers() const
        // Algorithm : Return player data without permitting changes outside the engine.
        {
            return state.players;
        }  //----- end of getPlayers


        ActionMask getActionMask(int playerNum) const
        // Algorithm : Return the legal moves cached in the current snapshot.
        {
            return state.players.at(playerNum).playerInfo.actionMask;
        }  //----- end of getActionMask

//---------------------------------------------- Constructors - destructor
        GameEngine(
            const GameConfig &p_config,
            const std::vector<Coord> &playersOrigins
        )
        // Algorithm : Validate configuration and initialize each snake before computing legal moves.
        {
            validateGameConfig(p_config);
            state.config = p_config;
            initPlayers(playersOrigins);
            updateActionMasks();
        }  //----- end of GameEngine

        explicit GameEngine(
            const GameState &snapshot
        ) : state(snapshot)
        // Algorithm : Copy the snapshot, validate its schedule and player data, then refresh derived masks.
        {
            validateGameConfig(state.config);

            if (state.round < 0 || state.currentPlayer < 1 || state.currentPlayer > state.config.N ||
                state.players.size() != static_cast<size_t>(state.config.N))
            {
                throw invalid_argument("Invalid game snapshot");
            }

            for (int player = 1; player <= state.config.N; player++)
            {
                const auto &snake = state.players.at(player).snake;

                if (snake.empty())
                {
                    throw invalid_argument("A player must retain at least one snake segment");
                }

                for (const Coord &cell : snake)
                {
                    if (!isInbound(cell, state.config))
                    {
                        throw invalid_argument("Snake segment outside the board");
                    }
                }

                auto &history = state.lastMoves[player];

                if (history.size() > ACTION_HISTORY_SIZE)
                {
                    throw invalid_argument("Action history exceeds its fixed size");
                }
            }

            updateActionMasks();
        }  //----- end of GameEngine

//---------------------------------------------------------------- PRIVATE
    private:
        void initPlayers(const std::vector<Coord> &playersOrigins);
        void updateSnake(int currentPlayer, const Coord &newHead);
        void updateActionMasks();
        bool isFree(const Coord &cell) const;

        GameState state;
        PlayerSelector selector;
};

#endif //GAMEENGINE_H
