/*************************************************************************
Minimax - Implementation of a minimax algorithm with alpha-beta prunning
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//---------- Interface of the module <Minimax> (file Minimax.h) ----------
#ifndef MINIMAX_H
#define MINIMAX_H

//------------------------------------------------------------------------
// Role of the <Minimax> module
// Implementation of a minimax algorithm with alpha-beta prunning
// that uses a reward function to evaluate the game state.
//------------------------------------------------------------------------

//-------------------------------------------------------- Used interfaces
#include "GameEngine.h"
#include "../include/Structs.h"
#include "../include/SpaceRiskAnalyzer.h"

//------------------------------------------------------------------ Types
// used to find the best optimised parameters
namespace std {
    template<> struct hash<GameConfig> {
        size_t operator()(const GameConfig& c) const {
            return hash<int>()(c.W) ^ hash<int>()(c.H) ^ hash<int>()(c.M);
        }
    };
}


// Reward configuration
struct Rewards {
    double selfSpace = 2.72685;  // default values in case the file cannot be read
    double oppSpace = 0.36796;

    double selfBlockRisk = -11.63637;
    double oppBlockRisk = -11.55511;

    double selfHeatRisk = -0.39336;
    double oppHeatRisk = -2.62494;

    double snakeLength = 1.66086;  // reward increase the more the player survives
};


struct MinimaxParams {
    Rewards rewards;
    int depth = 8;
};


//----------------------------------------------------------------- PUBLIC
class Minimax
{
//--------------------------------------------------------- Public Methods
    public:
        std::unordered_map<std::string, float> calculateMoveScores(
        	std::vector<float> &boardState,
            Players &playerState,
            int turn
        );
        // Usage :
        //
        // Contract :
        //

        Minimax(
            GameEngine p_gameEngine,
            int spaceRiskAnalyzerLookahead = 5,
            float spaceRiskAnalyzerDecay = 0.7
        ) : gameEngine(p_gameEngine),
            analyzer(p_gameEngine.getConfig(), spaceRiskAnalyzerLookahead, spaceRiskAnalyzerDecay)
        {
            #ifdef MAP
                cout << "Calling the constructor of Minimax" << endl;
            #endif
        }


    protected:
    	std::pair<std::vector<float>, Players> simulateMove(
        	const std::string &move,
            const std::vector<float> &boardState,
            const Players &playerState,
            int turn,
            int playerNum
        );
        // Usage :
        //
        // Contract :
        //

        float evaluateState(
            const Players &playerState,
            int currentPlayer
        );
        // Usage :
        //
        // Contract :
        //

        float minimax(
            const std::vector<float> &boardState,
            Players playerState,
            int depth,
            float alpha,
            float beta,
            int currentPlayer,
            int turn
        );
        // Usage :
        //
        // Contract :
        //


    private:
        GameEngine gameEngine;
        MinimaxParams params;
        SpaceRiskAnalyzer analyzer;
};

#endif //MINIMAX_H
