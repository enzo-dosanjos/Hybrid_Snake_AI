/*************************************************************************
Structs  -  Contains all the structures, types and constants used in the
            game's AI
                             -------------------
    copyright            : (C) 2025 by Enzo Dos Anjos
*************************************************************************/

//--------- Interface of the module <Structs> (file Structs.h) -----------
#ifndef STRUCTS_H
#define STRUCTS_H

//------------------------------------------------------------------------
// Role of the <Structs> module
// Contains all the structures, types and constants used in the game's AI
//------------------------------------------------------------------------

//---------------------------------------------------------------  INCLUDE
#include <deque>
#include <vector>
#include <unordered_map>
#include <array>


//------------------------------------------------------------------ Types
struct Coord {
    int x;
    int y;

    bool operator==(const Coord &other) const {
        return x == other.x && y == other.y;
    }
};

template <typename T>
class Matrix : public std::vector<std::vector<T>> {
    public:
        Matrix() = default;

        Matrix(int width, int height, T defaultValue = T())
            : std::vector<std::vector<T>>(height, std::vector<T>(width, defaultValue)) {}
};


// shared game structures
struct GameConfig {
    int W, H, M, N, P;

    bool operator==(const GameConfig &other) const {
        return W == other.W && H == other.H && M == other.M;  // doesn't consider the number of players and the player's turn
    }
};


struct PlayersInfo {
    Coord origin;
    bool alive;
    std::vector<bool> actionMask; // actions that the player can perform, indexed by ACTIONS

    bool operator==(const PlayersInfo &other) const {
        return alive == other.alive &&
               origin == other.origin;
    }
};

struct PlayersData {
    std::deque<Coord> snake; // snake body parts
    PlayersInfo playerInfo;
};

typedef std::unordered_map<int, PlayersData> Players;


//-------------------------------------------------------------- Constants
constexpr std::array<Coord, 4> DIRECTIONS = {
    Coord{-1, 0},  // left
    Coord{1, 0},  // right
    Coord{0, -1},  // up
    Coord{0, 1}  // down
};

const std::array<std::string, 4> ACTIONS = {
    "left",
    "right",
    "up",
    "down"
};

#endif //STRUCTS_H
