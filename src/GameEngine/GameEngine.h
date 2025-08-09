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
#include "../Utils.h"

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
        // Usage :
        //
        // Contract :
        //

        std::vector<bool> updateActionMask(
            const int &currentPlayer,
            const Coord &head
        );
        // Usage :
        //
        // Contract :
        //

        Players updateStep(
            const int &currentPlayer,
            const int &move
        );
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

        bool willDieNextTurn(
            const int &playerNum
        ) const;
        // Usage :
        //
        // Contract :
        //


        Players playerDied(
            const int &playerNum
        );


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

        Players &getPlayers()
        {
            return players;
        }

        std::vector<bool> getActionMask(const int &playerNum) const
        {
            return players.at(playerNum).playerInfo.actionMask;
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

//---------------------------------------------------------------- PRIVATE
    private:
        GameConfig config;
        int round, turn, numPlayerAlive;
        Players players;
};


#endif //GAMEENGINE_H
