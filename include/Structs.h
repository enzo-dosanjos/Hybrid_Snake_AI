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
#pragma once
#include <map>
#include <deque>
#include <vector>
#include <unordered_map>
#include <string>


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
        return W == other.W && H == other.H && M == other.M;
    }
};


struct PlayersInfo {
    Coord origin;
    bool alive;

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


struct InputMinMaxAvg {
    float min;
    float max;
    float avg;
};


//-------------------------------------------------------------- Constants
const std::map<std::string, Coord> DIRECTIONS = {
    {"left", Coord{-1, 0}},
    {"right", Coord{1, 0}},
    {"up", Coord{0, -1}},
    {"down", Coord{0, 1}}
};

#endif //STRUCTS_H
