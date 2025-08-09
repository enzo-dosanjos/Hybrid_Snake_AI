/*************************************************************************
GameEngine - Handles the game logic and operations
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//------- Interface of the module <GameEngine> (file GameEngine.h) -------
#ifndef GAMEENGINE_H
#define GAMEENGINE_H
#include <vector>

//------------------------------------------------------------------------
// Role of the <GameEngine> module
// The GameEngine module implements the game logic and operations.
//------------------------------------------------------------------------

//-------------------------------------------------------- Used interfaces
#include "SpaceRiskAnalyzer.h"
#include "Structs.h"

//-------------------------------------------------------------- Constants

//------------------------------------------------------------------ Types

//----------------------------------------------------------------- PUBLIC
class GameEngine
{
    public:
//--------------------------------------------------------- Public Methods
        Matrix<std::vector<float>> initBoard(
            const std::vector<Coord> &playersOrigins
        );
        // Usage :
        //
        // Contract :
        //

        Players initPlayers(
            const std::vector<Coord> &playersOrigins
        );
        // Usage :
        //
        // Contract :
        //

        Players snakeUpdate(
            const int &currentPlayer,
            const std::string &move
        );

        bool isInbound(
            int x,
            int y
        ) const;
        // Usage :
        //
        // Contract :
        //

        bool isDead(
            int playerNum
        ) const;
        // Usage :
        //
        // Contract :
        //

        bool terminalState();
        // Usage :
        //
        // Contract :
        //

        class PlayerSelector {
            public:
                void nextPlayer();

                PlayerSelector(
                ) : currentPlayer(1)
                {

                }

                int currentPlayer;
        };

        class Observation {
        public:
            Matrix<std::vector<float>> getObservation(
              const Players &players
            );
            // Usage :
            //
            // Contract :
            //

            static std::pair<int, int> getPresenceObservationsInd()
            // Algorithm : Get the indices of the presence observations in the board state
            {
                // The presence observations are the first 6 channels of the board state
                return {0, 6};
            }  // ----- End of getPresenceObservationsInd

            static int getPlayerHeadInd(
                int playerNum
            )
            // Algorithm : Get the index of the player's head in the board state
            {
                return playerNum == config.P ? 0 : 3;
            }

            static int getPlayerBodyInd(
                int playerNum
            )
            // Algorithm : Get the index of the player's body in the board state
            {
                return playerNum == config.P ? 1 : 4;
            }

            static int getPlayerTailInd(
                int playerNum
            )
            // Algorithm : Get the index of the player's tail in the board state
            {
                return playerNum == config.P ? 2 : 5;
            }

            Observation(
                const int p_boardChannels = 9
            ) : boardChannels(p_boardChannels)
            {

            }

            int boardChannels;
        };

        PlayerSelector playerSelector;
        Observation observation;

//---------------------------------------------- Constructors - destructor
        GameEngine(
            const GameConfig &p_config,
            const int p_boardChannels = 9,
            const std::vector<Coord> &playersOrigins
        ) : config(p_config), round(0), turn(0), numPlayerAlive(p_config.N),
            boardChannels(p_boardChannels), observation(p_boardChannels){
            initPlayers(playersOrigins);
            initBoard(playersOrigins);

        }

        ~GameEngine()
        {
        }

//-------------------------------------------------------------- PROTECTED

//---------------------------------------------------------------- PRIVATE
    private:
        GameConfig config;
        int round, turn, numPlayerAlive;
        Players players;

        int boardChannels;
        Matrix<std::vector<float>> board;
};



#endif //GAMEENGINE_H
