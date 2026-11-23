/*************************************************************************
SpaceRiskAnalyzer - Spatial evaluation of complete game snapshots
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//--- Implementation of the module <SpaceRiskAnalyzer> (file SpaceRiskAnalyzer.cpp) ---

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>

using namespace std;

//------------------------------------------------------- Personal Include
#include "SpaceRiskAnalyzer.h"
#include "../Utils.h"

//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
SpaceRiskResult SpaceRiskAnalyzer::analyzeState(const GameState &state, int player) const
// Algorithm : Build snapshot occupancy and reach data once, then derive territory and perspective risk.
{
    auto occupancy = precomputeOccupancy(state);

    // Full reach data serves territory; the heatmap applies its own lookahead cutoff.
    // Nothing is cached between calls, so simulated positions cannot reuse stale geometry.
    auto timing = computeReachTiming(state, occupancy.first, -1);
    auto areas = calculateAreas(state, occupancy.first, occupancy.second, timing);
    auto heatmap = calculateRiskHeatmap(timing, player);
    return {areas, heatmap};
}  //----- end of analyzeState


ReachTiming SpaceRiskAnalyzer::computeReachTiming(const GameState &state) const
// Algorithm : Build occupancy and explore each living player's paths up to the configured move horizon.
{
    auto occupancy = precomputeOccupancy(state);
    return computeReachTiming(state, occupancy.first, maxLookahead);
}  //----- end of computeReachTiming


AreaResult SpaceRiskAnalyzer::calculateAreas(const GameState &state) const
// Algorithm : Explore the complete static board and assign cells by shortest distance, leaving ties contested.
{
    auto occupancy = precomputeOccupancy(state);
    auto timing = computeReachTiming(state, occupancy.first, -1);
    return calculateAreas(state, occupancy.first, occupancy.second, timing);
}  //----- end of calculateAreas


Matrix<float> SpaceRiskAnalyzer::calculateRiskHeatmap(const ReachTiming &timing, int player) const
// Algorithm : Compare scheduled arrivals on reachable cells and apply the existing inverse-time risk and decay.
{
    if (timing.numPlayers <= 0 || player < 1 || player > timing.numPlayers || timing.slots.empty())
    {
        throw invalid_argument("Invalid reach timing or player perspective");
    }

    int height = timing.slots.size();
    int width = timing.slots.front().size();
    Matrix<float> heatmap(width, height, 0.f);

    for (int y = 0; y < height; y++)
    {
        if (timing.slots[y].size() != static_cast<size_t>(width))
        {
            throw invalid_argument("Reach timing must be rectangular");
        }

        for (int x = 0; x < width; x++)
        {
            const auto &arrivals = timing.slots[y][x];
            auto own = arrivals.find(player);

            // A cell with no own path contributes no directional pressure.
            if (own == arrivals.end())
            {
                continue;
            }

            // Ceil(slots / N) recovers move distance despite each player's turn offset.
            long long moves = (own->second + timing.numPlayers - 1) / timing.numPlayers;

            if (moves > maxLookahead)
            {
                continue;
            }

            float risk = 0.f;

            for (const auto &[opponent, arrival] : arrivals)
            {
                long long opponentMoves = (arrival + timing.numPlayers - 1) / timing.numPlayers;

                if (opponent == player || opponentMoves > maxLookahead)
                {
                    continue;
                }

                // Express time in round units to retain the old risk scale while resolving turn order.
                float difference = float(arrival - own->second) / timing.numPlayers;
                float opponentTime = float(arrival) / timing.numPlayers;
                risk += difference <= 0.f ? 1.f / (abs(difference) + 1.f) : 1.f / (opponentTime + 1.f);
            }

            heatmap[y][x] = risk * decayTable[moves];
        }
    }

    return heatmap;
}  //----- end of calculateRiskHeatmap


float SpaceRiskAnalyzer::calculateDirectionalRisk(const Matrix<float> &heatmap, Coord head) const
// Algorithm : Sum orthogonal neighbor risks using the supplied matrix dimensions rather than stored configuration.
{
    if (heatmap.empty() || heatmap.front().empty())
    {
        throw invalid_argument("Risk heatmap must not be empty");
    }

    int height = heatmap.size();
    int width = heatmap.front().size();

    for (const auto &row : heatmap)
    {
        if (row.size() != static_cast<size_t>(width))
        {
            throw invalid_argument("Risk heatmap must be rectangular");
        }
    }

    if (head.x < 0 || head.x >= width || head.y < 0 || head.y >= height)
    {
        throw invalid_argument("Head outside risk heatmap");
    }

    float risk = 0.f;

    for (const Coord &direction : DIRECTIONS)
    {
        int x = head.x + direction.x;
        int y = head.y + direction.y;

        if (x >= 0 && x < width && y >= 0 && y < height)
        {
            risk += heatmap[y][x];
        }
    }

    return risk;
}  //----- end of calculateDirectionalRisk


SpaceRiskAnalyzer::SpaceRiskAnalyzer(int lookahead, float decay)
    : maxLookahead(lookahead), decayFactor(decay)
// Algorithm : Validate the analysis horizon and decay before constructing the reusable decay table.
{
    if (lookahead < 0 || !isfinite(decay) || decay < 0.f || decay > 1.f)
    {
        throw invalid_argument("Invalid spatial analysis parameters");
    }

    precomputeDecay();
}  //----- end of SpaceRiskAnalyzer


//---------------------------------------------------------------- PRIVATE
pair<Matrix<bool>, Matrix<int>> SpaceRiskAnalyzer::precomputeOccupancy(const GameState &state) const
// Algorithm : Validate snapshot geometry and mark every retained segment, distinguishing living heads.
{
    const auto &config = state.config;
    validateGameConfig(config);

    if (state.currentPlayer < 1 || state.currentPlayer > config.N || state.round < 0 ||
        state.players.size() != static_cast<size_t>(config.N))
    {
        throw invalid_argument("Invalid spatial analysis snapshot");
    }

    Matrix<bool> occupied(config.W, config.H, false);
    Matrix<int> heads(config.W, config.H, 0);

    for (int player = 1; player <= config.N; player++)
    {
        const auto &data = state.players.at(player);

        if (data.snake.empty())
        {
            throw invalid_argument("Cannot analyze an empty snake");
        }

        // Dead bodies and tails remain obstacles; future tail release is not predicted here.
        for (const Coord &cell : data.snake)
        {
            if (!isInbound(cell, config))
            {
                throw invalid_argument("Snake segment outside the board");
            }

            occupied[cell.y][cell.x] = true;
        }

        if (data.playerInfo.alive)
        {
            Coord head = data.snake.front();
            heads[head.y][head.x] = player;
        }
    }

    return {occupied, heads};
}  //----- end of precomputeOccupancy


ReachTiming SpaceRiskAnalyzer::computeReachTiming(const GameState &state, const Matrix<bool> &occupied, int limit) const
// Algorithm : Run an independent breadth-first search per living head and translate path lengths into scheduled arrivals.
{
    const auto &config = state.config;
    ReachTiming timing{Matrix<unordered_map<int, long long>>(config.W, config.H), config.N};

    for (int player = 1; player <= config.N; player++)
    {
        const auto &data = state.players.at(player);

        if (!data.playerInfo.alive)
        {
            continue;
        }

        // Each player needs its own visited distances so shared cells retain all arrivals.
        // Paths may cross contested free cells; this estimates territory, not guaranteed control.
        Matrix<int> steps(config.W, config.H, -1);
        queue<Coord> pending;
        Coord head = data.snake.front();
        steps[head.y][head.x] = 0;
        timing.slots[head.y][head.x][player] = 0;
        pending.push(head);
        int offset = (player - state.currentPlayer + config.N) % config.N;

        while (!pending.empty())
        {
            Coord cell = pending.front();
            pending.pop();
            int distance = steps[cell.y][cell.x];

            if (limit >= 0 && distance >= limit)
            {
                continue;
            }

            for (const Coord &direction : DIRECTIONS)
            {
                Coord next{cell.x + direction.x, cell.y + direction.y};

                if (!isInbound(next, config) || occupied[next.y][next.x] || steps[next.y][next.x] >= 0)
                {
                    continue;
                }

                steps[next.y][next.x] = distance + 1;
                // The next player's first move finishes at slot 1; dead slots still occupy time.
                timing.slots[next.y][next.x][player] = offset + 1LL + static_cast<long long>(distance) * config.N;
                pending.push(next);
            }
        }
    }

    return timing;
}  //----- end of computeReachTiming


AreaResult SpaceRiskAnalyzer::calculateAreas(const GameState &state, const Matrix<bool> &occupied,
                                             const Matrix<int> &heads, const ReachTiming &timing) const
// Algorithm : Assign nearest-player territory and average the original adjacent-obstacle pressure over owned cells.
{
    const auto &config = state.config;
    AreaResult areas{vector<float>(config.N + 1, 0.f), vector<float>(config.N + 1, 0.f)};

    for (int y = 0; y < config.H; y++)
    {
        for (int x = 0; x < config.W; x++)
        {
            int owner = 0;
            long long shortest = numeric_limits<long long>::max();

            for (const auto &[player, arrival] : timing.slots[y][x])
            {
                // Territory uses move distance, deliberately discarding scheduled turn priority.
                long long distance = (arrival + config.N - 1) / config.N;

                if (distance < shortest)
                {
                    shortest = distance;
                    owner = player;
                }
                else if (distance == shortest)
                {
                    owner = 0;
                }
            }

            // Unreachable cells and equal-distance contested cells contribute to neither player.
            if (owner == 0)
            {
                continue;
            }

            areas.space[owner]++;
            float pressure = 0.f;
            int directions = 0;

            for (const Coord &direction : DIRECTIONS)
            {
                Coord next{x + direction.x, y + direction.y};

                if (!isInbound(next, config))
                {
                    continue;
                }

                directions++;

                if (occupied[next.y][next.x])
                {
                    pressure += 1.f;
                    int head = heads[next.y][next.x];

                    // Retain the existing additional pressure from own and opponent living heads.
                    if (head != 0)
                    {
                        pressure += head == owner ? 0.5f : 2.f;
                    }
                }
            }

            if (directions > 0)
            {
                areas.blockRisk[owner] += pressure / directions;
            }
        }
    }

    // These are heuristic pressure values, not probabilities; head weights can exceed one.
    for (int player = 1; player <= config.N; player++)
    {
        if (areas.space[player] > 0.f)
        {
            areas.blockRisk[player] /= areas.space[player];
        }
    }

    return areas;
}  //----- end of calculateAreas


void SpaceRiskAnalyzer::precomputeDecay()
// Algorithm : Precompute exponential risk decay once for each allowed move distance.
{
    decayTable.resize(static_cast<size_t>(maxLookahead) + 1);

    for (int distance = 0; distance <= maxLookahead; distance++)
    {
        decayTable[distance] = pow(decayFactor, distance);
    }
}  //----- end of precomputeDecay
