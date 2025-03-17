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
        bool isInbound(
            int x,
            int y
        );
        // Usage :
        //
        // Contract :
        //

        bool checkDeath(
            const PlayersData &playerState,
            const std::vector<float> &boardState,
            int playerNb
        );
        // Usage :
        //
        // Contract :
        //

        bool checkTerminalState(
            std::vector<float> boardState,
            PlayersData playerState
        );
        // Usage :
        //
        // Contract :
        //

        std::vector<std::string> getValidMoves(
          const std::vector<float> &boardState,
          const PlayersData &playerState,
          int playerNb
        );
        // Usage :
        //
        // Contract :
        //

        int getNextPlayer(
          int currentPlayer,
          const PlayersData &playerState
        );
        // Usage :
        //
        // Contract :
        //

        std::vector<float> computeBoardState(
          const PlayersData &Players
        );
        // Usage :
        //
        // Contract :
        //

        std::vector<float> computeExtraFeatures(
          const std::vector<float> boardState,
          const PlayersData &playerState,
          std::unordered_map<int, std::vector<std::string>> validMoves,
          const std::deque<std::string> &last5Moves,
          int turn
        );
        // Usage :
        //
        // Contract :
        //

        float calculateKillZoneProximity(
          Coord opp_head
        );
        // Usage :
        //
        // Contract :
        //

        float calculateActionEntropy(
          const std::deque<std::string>& moves
        );
        // Usage :
        //
        // Contract :
        //


//---------------------------------------------- Constructors - destructor
        Utils(
            int w,
            int h,
            int m,
            int n,
            int p,
            int boardChannels
        ) : config({w, h, m, n, p}), boardChannels(boardChannels)
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
