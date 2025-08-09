/*************************************************************************
Utils - utility functions for the game processing
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//------------ Interface of the module <Utils> (file Utils.h) ------------
#ifndef UTILS_H
#define UTILS_H

//------------------------------------------------------------------------
// Role of the <Utils> module
// This module contains utility functions for the game processing.
//------------------------------------------------------------------------

//-------------------------------------------------------- Used interfaces
#include "Structs.h"


//----------------------------------------------------------------- PUBLIC
class Utils
{
//--------------------------------------------------------- Public Methods
    public:
        std::vector<std::string> getValidMoves(
          const std::vector<float> &boardState,  // todo: change
          const Players &playerState,
          int playerNum
        );
        // Usage :
        //
        // Contract :
        //

        std::vector<float> computeExtraFeatures(
          const std::vector<float> boardState,  // todo: change
          const Players &playerState,
          std::unordered_map<int, std::vector<std::string>> validMoves,
          const std::deque<std::string> &last5Moves,
          int turn
        );
        // Usage :
        //
        // Contract :
        //

        float calculateKillZoneProximity(
          Coord oppHead
        ) const;
        // Usage :
        //
        // Contract :
        //

        static float calculateActionEntropy(
          const std::deque<std::string>& moves
        );
        // Usage :
        //
        // Contract :
        //

        Matrix<std::vector<float>> reshapeBoardState(
                const std::vector<float> &flatBoardState
            ) const;
        // Usage :
        //
        // Contract :
        //


//---------------------------------------------- Constructors - destructor
        Utils(
            const GameConfig &p_config,
            int boardChannels
        ) : config(p_config), boardChannels(boardChannels)
        // Usage :
        //
        // Contract :
        //
        {
            #ifdef MAP
                cout << "Calling the constructor of Utils" << endl;
            #endif
        };


//---------------------------------------------------------------- PRIVATE
    private:
        GameConfig config;
        int boardChannels;
};



#endif //UTILS_H
