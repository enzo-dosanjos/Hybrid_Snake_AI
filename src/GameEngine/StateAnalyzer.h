/*************************************************************************
StateAnalyzer - todo
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//---- Interface of the module <StateAnalyzer> (file StateAnalyzer.h) ----
#ifndef STATEANALYZER_H
#define STATEANALYZER_H

//------------------------------------------------------------------------
// Role of the <StateAnalyzer> module
// todo
//------------------------------------------------------------------------

//-------------------------------------------------------- Used interfaces
#include "GameEngine.h"
#include "Structs.h"

//----------------------------------------------------------------- PUBLIC
class StateAnalyzer
{
//--------------------------------------------------------- Public Methods
    public:
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

//---------------------------------------------- Constructors - destructor
        StateAnalyzer(
            const GameEngine &p_gameEngine
        ) : gameEngine(p_gameEngine)
        {
            #ifdef MAP
                        cout << "Calling the constructor of StateAnalyzer" << endl;
            #endif
        };

    private:
        GameEngine gameEngine;
};



#endif //STATEANALYZER_H
