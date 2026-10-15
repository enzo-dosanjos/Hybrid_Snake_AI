/*************************************************************************
InputHandler - Parses configuration, origins and complete events
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//-- Implementation of the module <InputHandler> (file InputHandler.cpp) --

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <sstream>
#include <stdexcept>

using namespace std;

//------------------------------------------------------- Personal Include
#include "InputHandler.h"
#include "../Utils.h"

//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
GameConfig InputHandler::readGameConfig(istream &input)
// Algorithm : Read configuration fields and reject invalid rules before allocating player data.
{
    GameConfig config;

    if (!(
        input >> config.W >> config.H >>    // size of the board
        config.M >>                         // grow 1 block every M rounds
        config.N >> config.P))              // number of players and player's turn
    {
        throw invalid_argument("Incomplete or malformed game configuration");
    }

    validateGameConfig(config);
    return config;
}  //----- end of readGameConfig


vector<Coord> InputHandler::readPlayersOrigins(int numPlayers, istream &input)
// Algorithm : Read one origin per player while reserving index zero for the one-based convention.
{
    if (numPlayers <= 0)
    {
        throw invalid_argument("Invalid player count");
    }

    vector<Coord> origins(static_cast<size_t>(numPlayers) + 1);

    for (int player = 1; player <= numPlayers; player++)
    {
        if (!(input >> origins[player].x >> origins[player].y))
        {
            throw invalid_argument("Incomplete or malformed player origins");
        }
    }

    return origins;
}  //----- end of readPlayersOrigins


GameEvent InputHandler::readEvent(int numPlayers, istream &input)
// Algorithm : Parse one event line, normalize its player ID and reject invalid tokens or trailing data.
{
    GameEvent event;
    string line;

    if (!getline(input >> ws, line))
    {
        if (!input.eof() || input.bad())
        {
            event.type = GameEventType::InputError;
            event.error = "Cannot read game event";
        }

        return event;
    }

    event.type = GameEventType::InputError;
    event.error = "Malformed game event: " + line;
    istringstream message(line);
    string kind;
    int externalPlayer;

    if (!(message >> kind >> externalPlayer) || externalPlayer < 0 || externalPlayer >= numPlayers)
    {
        return event;
    }

    event.player = externalPlayer + 1;

    if (kind == "move")
    {
        string direction;

        if (!(message >> direction))
        {
            return event;
        }

        for (size_t action = 0; action < ACTIONS.size(); action++)
        {
            if (ACTIONS[action] == direction)
            {
                event.action = static_cast<Action>(action);
                break;
            }
        }

        if (!event.action)
        {
            return event;
        }
    }
    else if (kind != "death")
    {
        return event;
    }

    string extra;

    if (message >> extra)
    {
        return event;
    }

    event.type = kind == "move" ? GameEventType::Move : GameEventType::Death;
    event.error.clear();
    return event;
}  //----- end of readEvent


void InputHandler::writeMove(Action move, ostream &output)
// Algorithm : Write the action's protocol name and flush the completed line.
{
    output << ACTIONS.at(static_cast<int>(move)) << endl;
}  //----- end of writeMove
