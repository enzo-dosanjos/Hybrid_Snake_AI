/*************************************************************************
GameEngine - Handles the game logic and operations
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//------- Interface of the module <GameEngine> (file GameEngine.h) -------
#ifndef GAMEENGINE_H
#define GAMEENGINE_H

//------------------------------------------------------------------------
// Role of the <GameEngine> module
// The GameEngine module implements the game logic and operations.
//------------------------------------------------------------------------

//-------------------------------------------------------- Used interfaces
#include "../Structs.h"

//-------------------------------------------------------------- Constants

//------------------------------------------------------------------ Types

//----------------------------------------------------------------- PUBLIC
class GameEngine
{
//--------------------------------------------------------- Public Methods
    public:
        Players initPlayers(
            const std::vector<Coord> &playersOrigins
        );
        // Usage :
        //
        // Contract :
        //

        Players updateSnake(
            const int &currentPlayer,
            const Coord &newHead
        );

        std::vector<bool> updateActionMask(
            const int &currentPlayer,
            const Coord &newHead
        );

        Players updateStep(
            const int &currentPlayer,
            const int &move
        );

        bool isInbound(
            Coord coord
        ) const;
        // Usage :
        //
        // Contract :
        //

        bool isAlive(
            const int &playerNum,
            const Coord &head = {-1, -1}
        ) const;
        // Usage :
        //
        // Contract :
        //

        bool isTerminalState();
        // Usage :
        //
        // Contract :
        //

//--------------------------------------------------------- SETTERS/GETTERS
        GameConfig getConfig() const
        {
            return config;
        }

        int getW() const
        {
            return config.W;
        }

        int getH() const
        {
            return config.H;
        }

        int getM() const
        {
            return config.M;
        }

        int getN() const
        {
            return config.N;
        }

        int getP() const
        {
            return config.P;
        }

        int getRound() const
        {
            return round;
        }

        int getTurn() const
        {
            return turn;
        }

        int getNumPlayerAlive() const
        {
            return numPlayerAlive;
        }

        Players getPlayers() const
        {
            return players;
        }

//---------------------------------------------- Constructors - destructor
        GameEngine(
            const GameConfig &p_config,
            const std::vector<Coord> &playersOrigins
        ) : config(p_config), round(0), turn(0), numPlayerAlive(p_config.N)
        {
            initPlayers(playersOrigins);
        }

        ~GameEngine()
        {
        }

//-------------------------------------------------------------- PROTECTED
    protected:
        static Coord computeNewCoord(
            const Coord &currentHead,
            const int &move
        );
        // Usage :
        //
        // Contract :
        // Computes the new coordinates of the snake's head based on the current head and the move

//---------------------------------------------------------------- PRIVATE
    private:
        GameConfig config;
        int round, turn, numPlayerAlive;
        Players players;
};


#endif //GAMEENGINE_H
