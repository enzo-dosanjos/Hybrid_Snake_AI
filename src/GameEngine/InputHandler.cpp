/*************************************************************************
InputHandler - todo
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//-- Implementation of the module <InputHandler> (file InputHandler.cpp) --

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <iostream>

using namespace std;

//------------------------------------------------------- Personal Include
#include "InputHandler.h"

//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
GameConfig InputHandler::readGameConfig()
// Algorithm : Read the game configuration from the input
{
    int W, H, M, N, P;

    cin >> W >> H;  // size of the board
    cin >> M;       // grow 1 block every M rounds
    cin >> N >> P;  // number of players and player's turn

    return GameConfig{W, H, M, N, P};
}  // ----- End of readGameConfig

vector<Coord> InputHandler::readPlayersOrigins(int numPlayers)
{
    vector<Coord> playersOrigins(numPlayers + 1);  // player origins, index 0 is unused
    int X, Y;

    for (int i = 1; i <= numPlayers; i++) {
        cin >> X >> Y;
        playersOrigins[i] = Coord{X, Y};
    }

    return playersOrigins;
}  // ----- End of readPlayersOrigins

pair<string, int> InputHandler::readAction()
// Algorithm : Read the action from the input
{
    string action;
    int playerNb;

    cin >> action >> playerNb;

    return {action, playerNb};
}  // ----- End of readAction

int InputHandler::readMove()
// Algorithm : Read the move from the input
{
    string moveStr;
    int move = -1;

    cin >> moveStr;

    for (int i = 0; i < ACTIONS.size(); i++) {
        if (ACTIONS[i] == moveStr) {
            move = i;
            break;
        }
    }

    return move;
}  // ----- End of readMove

void InputHandler::processDeathEvent()
// Algorithm : Process the death event from the input
{
    return;
}

void InputHandler::writeMove(const int move)
// Algorithm : Write the move to the output
{
    string moveStr = ACTIONS[move];
    cout << moveStr << endl;
}