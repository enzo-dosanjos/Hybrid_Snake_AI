/*************************************************************************
SpaceRiskAnalyzer - A spatial risk assessment system for grid-based games
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

// Implementation of the module <SpaceRiskAnalyzer> (file SpaceRiskAnalyzer.cpp)

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <iostream>
#include <cmath>
#include <queue>

using namespace std;

//------------------------------------------------------- Personal Include
#include "../include/SpaceRiskAnalyzer.h"

//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
pair<pair<vector<float>, vector<float>>, Matrix<float>> SpaceRiskAnalyzer::analyzeState(const PlayersData &playerState,
                                                                                   int currentPlayer)
// Algorithm :
{
    // check if computing occupancy and reach data is necessary
    if(playerState != lastPlayerState) {
        pair <vector<vector<bool>>, vector<vector<int>>> occupancy = precomputeOccupancy(playerState);
        lastOccupied = occupancy.first;
        lastHeads = occupancy.second;

        lastReachData = computeReachTiming(playerState, lastOccupied);
        lastPlayerState = playerState;
    }

    // Calculate results using shared precomputed data
    pair<vector<float>, vector<float>> areas = calculateAreas(playerState, lastOccupied, lastHeads);
    vector<vector<float>> heatmap = calculateRiskHeatmap(playerState, lastOccupied, currentPlayer);

    return {areas, heatmap};
}  // ----- End of analyzeState


Matrix<float> SpaceRiskAnalyzer::calculateRiskHeatmap(const PlayersData &playerState,
                                                      const vector<vector<bool>> &occupied, int currentPlayer)
// Algorithm : For every cell, calculate the risk factor for the current player based on the timing
// taken by other players to reach the cell. An exponential decay is applied the further the lookahead
{
    vector<vector<float>> heatmap(H, vector<float>(W, 0.0));

    vector<vector<int>> currentTiming = precomputeCurrentPlayerTiming(lastReachData, currentPlayer);

    for (int y = 0; y < H; y++)
    {
        for (int x = 0; x < W; x++)
        {
            float cellRisk = 0.0;

            // Get timing for the current player
            int timing = currentTiming[y][x];

            for (const auto &timings : lastReachData[y][x])
            {
                // for every timing took for other players to reach the cell, compute the difference
                // and add a risk factor inversely proportional to the difference
                if (timings.first != currentPlayer)
                {
                    float diff = timings.second - timing;
                    cellRisk += (diff <= 0) ? 1.0 / (abs(diff) + 1) : 1.0 / (timings.second + 1);
                }
            }

            // Apply exponential decay over lookahead steps
            heatmap[y][x] = cellRisk * decayTable[min(timing, maxLookahead)];
        }
    }

    return heatmap;
}  // ----- End of calculateRiskHeatmap


pair<vector<float>, vector<float>> SpaceRiskAnalyzer::calculateAreas(const PlayersData &playerState,
                                                            const Matrix<bool> &occupied, const Matrix<int> &heads)
// Algorithm : Compute the accessible area of each players using a multi-source flood fill algorithm.
// During the flood fill, the risk of being blocked by another player is computed by counting the
// number of movements blocked by other players.
{
    int nbPlayer = playerState.size();

    // matrix of the board containing:
    // 0   : cell not reached
    // i   : cell reached only by player i
    // -1  : cell reached by multiple player
    vector<vector<int>> visited(H, vector<int>(W, 0));
    vector<deque<Coord>> queues(nbPlayer + 1);

    // Keep count of the number of reachable cells for each players
    vector<float> space(nbPlayer + 1, 0.0);
    // Keep count of the neighbooring cells that could kill the player during the exploration
    vector<float> blockRisk(nbPlayer + 1, 0.0);

    // initialise the queues with the player's head
    for (int i = 1; i < nbPlayer + 1; i++)
    {
        Coord playerHead = playerState.at(i).first[0];
        queues[i].push_back(playerHead);
        visited[playerHead.second][playerHead.first] = i;
    }


    bool progress = true;  // check if there's any progress (at least a neighboor is neither occupied nor already visited)
    while (progress)
    {
        progress = false;
        for (int i = 1; i < nbPlayer + 1; i++)
        {
            deque<Coord> newQueue;  // create a temporary queue to avoid adding cells that will become contested
            while (!queues[i].empty())
            {
                int x = queues[i].front().first;
                int y = queues[i].front().second;

                queues[i].pop_front();  // remove the cell from the queue

                if (visited[y][x] != i)
                {
                    continue;  // if the cell was already reached by another player, we don't explore its neighboors
                }

                // For each neighbooring cells
                float playerBlockRisk = 0.0;
                int validDirections = 0;
                for (auto direction : DIRECTIONS)
                {
                    int newX = x + direction.second.first;
                    int newY = y + direction.second.second;

                    if (helper.isInbound(newX, newY))
                    {
                        validDirections++;
                        // if the cell is not occupied by another snake
                        if (!occupied[newY][newX])
                        {

                            // if the cell was never reached
                            if (visited[newY][newX] == 0)
                            {
                                visited[newY][newX] = i;
                                newQueue.push_back({newX, newY});
                                progress = true;
                            }
                            // if the cell was already reached by at most one other player
                            else if (visited[newY][newX] != i && visited[newY][newX] != -1)
                            {
                                visited[newY][newX] = -1;
                                // the cell is now contested so we don't add it to the queue
                                progress = true;
                            }
                        }
                        // otherwise, it is a risk to get blocked by another player
                        else
                        {
                            playerBlockRisk += 1.0;
                            if (heads[newY][newX] != 0) {
                                // Higher risk for opponent heads
                                playerBlockRisk += (heads[newY][newX] == i) ? 0.5f : 2.0f;
                            }
                        }
                    }
                }

                if (validDirections > 0)
                {
                    blockRisk[i] += playerBlockRisk / validDirections;   // Count adjacent blocked cells normalised
                                                                         // by possible directions
                }
            }

            queues[i] = newQueue;
        }
    }

    // Count the reachable space of each player by checking the cells with their number
    // the cells with -1 (contested cells) aren't counted as reachable
    for (int y = 0; y < H; y++)
    {
        for (int x = 0; x < W; x++)
        {
            if (visited[y][x] > 0)
            {
                int p = visited[y][x];
                space[p] += 1.0;
            }
        }
    }

    for (int i = 1; i < nbPlayer + 1; i++)
    {
        if (space[i] > 0)
        {
            blockRisk[i] /= space[i];  // blockRisk take into account the available space to avoid moments
                                       // when there's only one space available so the blockRisk is one
        }
        else
        {
            blockRisk[i] = 1.0;
        }
    }

    return {space, blockRisk};
}  // ----- End of calculateAreas


float SpaceRiskAnalyzer::calculateDirectionalRisk(const Matrix<float> &heatmap, Coord head)
// Algorithm : For a given cell, calculate the risk of moving in each direction by summing the risk
// of the heatmap in the neighbooring cells
{
    float risk = 0.0;

    // Evaluate risk for every directions
    for (const auto &direction : DIRECTIONS)
    {
        int newX = head.first + direction.second.first;
        int newY = head.second + direction.second.second;

        if (helper.isInbound(newX, newY))
        {
            risk += heatmap[newY][newX];
        }
    }
    return risk;
}  // ----- End of calculateDirectionalRisk


//-------------------------------------------------------------- PROTECTED
Matrix<unordered_map<int, int>> SpaceRiskAnalyzer::computeReachTiming(const PlayersData &playerState,
                                                                      const Matrix<bool> &occupied)
// Algorithm : Compute the timing grid for all players using a single BFS. The timing grid is a matrix
// of unordered_map where each cell contains the timing taken by each player to reach the cell.
// The BFS is stopped when the lookahead is reached.
{

    Matrix<unordered_map<int, int>> timingGrid(H, vector<unordered_map<int, int>>(W));
    Matrix<int> steps(H, vector<int>(W, -1));

    queue<Coord> queue;  // Single BFS for all players to speed up the computation

    // Initialize with all player heads
    for (const auto& player : playerState)
    {
        Coord head = player.second.first.front();

        if (steps[head.second][head.first] == -1)
        {
            steps[head.second][head.first] = 0;
            timingGrid[head.second][head.first][player.first] = 0;

            queue.push(head);
        }
    }

    // Unified BFS
    while (!queue.empty())
    {
        int x = queue.front().first;
        int y = queue.front().second;
        queue.pop();

        for (const auto& dir : DIRECTIONS)
        {
            int newX = x + dir.second.first;
            int newY = y + dir.second.second;

            // check if the cell is inbound, not occupied and not visited
            if (helper.isInbound(newX, newY) && !occupied[newY][newX] && steps[newY][newX] == -1)
            {
                steps[newY][newX] = steps[y][x] + 1;  // add 1 to steps needed to reach the parent cell

                // Propagate all players' timings
                for (const auto &entry : timingGrid[y][x])
                {
                    timingGrid[newY][newX][entry.first] = steps[newY][newX];
                }

                // add the neighboors if the lookahead is not reached
                if (steps[newY][newX] < maxLookahead)
                {
                    queue.push({newX, newY});
                }
            }
        }
    }

    return timingGrid;
}  // ----- End of computeReachTiming


Matrix<int> SpaceRiskAnalyzer::precomputeCurrentPlayerTiming( const Matrix<unordered_map<int, int>> &timingGrid,
                                                             int currentPlayer)
// Algorithm : Precompute the timing grid for the current player
{
    Matrix<int> currentTiming(H, vector<int>(W, maxLookahead + 1));

    for (int y = 0; y < H; y++)
    {
        for (int x = 0; x < W; x++)
        {
            unordered_map<int, int>::const_iterator it = timingGrid[y][x].find(currentPlayer);
            currentTiming[y][x] = (it != end(timingGrid[y][x])) ? it->second : maxLookahead + 1;
        }
    }
    return currentTiming;
}  // ----- End of precomputeCurrentPlayerTiming


pair<Matrix<bool>, Matrix<int>> SpaceRiskAnalyzer::precomputeOccupancy(const PlayersData &playerState)
// Algorithm : Precompute the occupied cells and heads for all players to avoid recomputing
// the same data for calculateAreas and calculateRiskHeatmap
{
    // Precompute occupied cells and heads
    Matrix<bool> occupied(H, vector<bool>(W, false));
    Matrix<int> heads(H, vector<int>(W, 0));

    // Fill the matrices with every player's body parts
    for (const auto &player : playerState)
    {
        for (const auto &part : player.second.first)
        {
            occupied[part.second][part.first] = true;
        }

        Coord head = player.second.first.front();
        heads[head.second][head.first] = player.first;
    }

    return {occupied, heads};
}  // ----- End of precomputeOccupancy


void SpaceRiskAnalyzer::precomputeDecay()
// Algorithm : Precompute the decay table for the exponential decay of the risk heatmap
// to avoid recomputing it at each step
{
    decayTable.resize(maxLookahead + 1);

    for (int i = 0; i <= maxLookahead; i++)
    {
        decayTable[i] = pow(decayFactor, i);
    }
}  // ----- End of precomputeDecay
