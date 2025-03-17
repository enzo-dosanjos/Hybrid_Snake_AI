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
typedef std::pair<int, int> Coord;

template <typename T>
using Matrix = std::vector<std::vector<T>>;


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


typedef std::unordered_map<int, std::pair<std::deque<Coord>, PlayersInfo>> PlayersData;


struct InputMinMaxAvg {
    float min;
    float max;
    float avg;
};


//-------------------------------------------------------------- Constants
const std::map<std::string, Coord> DIRECTIONS = {
    {"left", std::make_pair(-1, 0)},
    {"right", std::make_pair(1, 0)},
    {"up", std::make_pair(0, -1)},
    {"down", std::make_pair(0, 1)}
};

#endif //STRUCTS_H
