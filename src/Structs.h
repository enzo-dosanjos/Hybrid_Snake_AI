/*************************************************************************
Structs  -  Contains all the shared structures, types and constants and
            data contracts
                             -------------------
    copyright            : (C) 2025 by Enzo Dos Anjos
*************************************************************************/

//--------- Interface of the module <Structs> (file Structs.h) -----------
#ifndef STRUCTS_H
#define STRUCTS_H

//------------------------------------------------------------------------
// Role of the <Structs> module
// Defines values shared by the engine, protocol and AI modules.
//------------------------------------------------------------------------

//---------------------------------------------------------------  INCLUDE
#include <array>
#include <bitset>
#include <deque>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

//------------------------------------------------------------------ Types
enum class Action {
    Left = 0,
    Up = 1,
    Right = 2,
    Down = 3
};

using ActionMask = std::bitset<4>;

struct Coord {
    int x = 0;
    int y = 0;

    bool operator==(const Coord &other) const
    // Algorithm : Compare both board coordinates.
    {
        return x == other.x && y == other.y;
    }  //----- end of operator==
};


template <typename T>
class Matrix : public std::vector<std::vector<T>> {
    public:
        Matrix() = default;

        Matrix(int width, int height, T defaultValue = T())
            : std::vector<std::vector<T>>(height, std::vector<T>(width, defaultValue))
        // Algorithm : Allocate a matrix indexed by row y, then column x.
        {
        }  //----- end of Matrix
};


struct GameConfig {
    int W = 0;
    int H = 0;
    int M = 0;
    int N = 0;
    int P = 0;

    bool operator==(const GameConfig &other) const
    // Algorithm : Compare board rules, player count and controlled-player identity.
    {
        return W == other.W && H == other.H && M == other.M &&
               N == other.N && P == other.P;
    }  //----- end of operator==
};


struct PlayersInfo {
    Coord origin;
    bool alive = false;
    ActionMask actionMask;   // actions that the player can perform

    bool operator==(const PlayersInfo &other) const
    // Algorithm : Compare the player's origin, alive flag and derived legal moves.
    {
        return alive == other.alive && origin == other.origin && actionMask == other.actionMask;
    }  //----- end of operator==
};


struct PlayersData {
    std::deque<Coord> snake;  // The head is at the front and the tail is at the back.
    PlayersInfo playerInfo;

    bool operator==(const PlayersData &other) const
    // Algorithm : Compare all snake segments and player metadata.
    {
        return snake == other.snake && playerInfo == other.playerInfo;
    }  //----- end of operator==
};

// Keys use one-based player IDs, including players that have died.
using Players = std::unordered_map<int, PlayersData>;


// State values own their containers; copying a snapshot creates an independent branch.
struct GameState {
    GameConfig config;
    Players players;
    int currentPlayer = 1;
    // Round zero grows; dead-player slots still count toward each round.
    int round = 0;
    std::map<int, std::deque<Action>> lastMoves;
};


enum class GameEventType {
    Move,
    Death,
    End,
    InputError
};

struct GameEvent {
    GameEventType type = GameEventType::End;
    // Player IDs are normalized to 1..N at the input boundary.
    int player = 0;
    std::optional<Action> action;
    std::string error;
};


// Neural-network data uses channels first; each channel is a [y][x] matrix.
using BoardTensor = std::vector<Matrix<float>>;
using FeatureTensor = std::vector<Matrix<float>>;

struct EncodedState {
    BoardTensor board;
    std::vector<float> extraFeatures;
    ActionMask legalActions;
};

struct Transition {
    EncodedState state;
    Action action = Action::Left;
    float reward = 0.f;
    EncodedState nextState;
    bool done = false;
};

constexpr int EXTRA_FEATURE_SCHEMA_VERSION = 1;  // Alows to identify which layout and meaning of extra features a model expects

struct ModelConfig {
    int numPlayers = 2;
    int featureSchemaVersion = EXTRA_FEATURE_SCHEMA_VERSION;
    // Set from StateAnalyzer::getFeatureCount(numPlayers) when constructing a model.
    int extraFeatureCount = 0;
    int replayCapacity = 100000;
    float gamma = 0.99f;
    float epsilon = 1.f;
    float minEpsilon = 0.01f;
    float epsilonDecay = 0.995f;
};


//-------------------------------------------------------------- Constants
constexpr std::array<Coord, 4> DIRECTIONS = {
    Coord{-1, 0},  // left
    Coord{0, -1},  // up
    Coord{1, 0},  // right
    Coord{0, 1}  // down
};

const std::array<std::string, 4> ACTIONS = {
    "left",
    "up",
    "right",
    "down"
};

constexpr std::size_t ACTION_HISTORY_SIZE = 5;
constexpr std::size_t BOARD_CHANNELS = 9;

#endif //STRUCTS_H
