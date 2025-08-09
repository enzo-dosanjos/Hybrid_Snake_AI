/*************************************************************************
Observation - Manages the board representation of the game
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//------ Interface of the module <Observation> (file Observation.h) ------
#ifndef OBSERVATION_H
#define OBSERVATION_H


//------------------------------------------------------------------------
// Role of the <Observation> module
// This module is responsible for managing the board representation used
// by the AI's CNN. It provides methods to initialize and update the board
// state based on the players' positions and actions.
//------------------------------------------------------------------------

//-------------------------------------------------------- Used interfaces
#include "../Structs.h"
#include "../Utils.h"

//-------------------------------------------------------------- Constants

//------------------------------------------------------------------ Types

//----------------------------------------------------------------- PUBLIC
class Observation {
//--------------------------------------------------------- Public Methods
    public:
        Matrix<std::vector<float>> getObservation() const;
        // Usage :
        //
        // Contract :
        //

        Matrix<std::vector<float>> initBoard(
            const std::vector<Coord> &playersOrigins
        );
        // Usage :
        //
        // Contract :
        //

        Matrix<std::vector<float>> updateBoard(
            const Players &playerState,
            const int &currentPlayer,
            const int &round
        );

        static std::pair<int, int> getPresenceObservationsInd()
        // Algorithm : Get the indices of the presence observations in the board state
        {
            // The presence observations are the first 6 channels of the board state
            return {0, 6};
        }  // ----- End of getPresenceObservationsInd

        std::pair<int, int> getPlayerHeadInd(
            int playerNum
        )
        // Algorithm : Get the index of the player's head in the board state
        {
            return playerNum == config.P ? std::make_pair(0, 0) : std::make_pair(3, 3);
        }

        std::pair<int, int> getPlayerBodyInd(
            int playerNum
        )
        // Algorithm : Get the index of the player's body in the board state
        {
            return playerNum == config.P ? std::make_pair(1, 1) : std::make_pair(4, 4);
        }

        std::pair<int, int> getPlayerTailInd(
            int playerNum
        )
        // Algorithm : Get the index of the player's tail in the board state
        {
            return playerNum == config.P ? std::make_pair(2, 2) : std::make_pair(5, 5);
        }

        static std::pair<int, int> getAccessibleCellInd()
        // Algorithm : Get the index of the accessible cell in the board state
        {
            return {6, 6};
        }

        static std::pair<int, int> getMyHeadXInd()
        // Algorithm : Get the index of my snake head x coordinate in the board state
        {
            return {7, 7};
        }

        static std::pair<int, int> getMyHeadYInd()
        // Algorithm : Get the index of my snake head y coordinate in the board state
        {
            return {8, 8};
        }

//---------------------------------------------- Constructors - destructor
        Observation(
            const GameConfig &p_config,
            const std::vector<Coord> &playersOrigins,
            const int p_boardChannels = 9
        ) : config(p_config), boardChannels(p_boardChannels)
        {
            #ifdef MAP
                std::cout << "Calling the constructor of Observation" << std::endl;
            #endif

            initBoard(playersOrigins);
        }

//--------------------------------------------------------------- PROTECTED
    protected:
        Coord findPrevSegment(
            const Coord& segment,
            const std::pair<int, int>& inds
        ) const;

//---------------------------------------------------------------- PRIVATE
    private:
        GameConfig config;

        int boardChannels;
        Matrix<std::vector<float>> board;

};



#endif //OBSERVATION_H
