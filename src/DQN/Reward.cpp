/*************************************************************************
Reward - Scores the interval between learner decisions
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//-------- Implementation of the module <Reward> (file Reward.cpp) ---------

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <algorithm>
#include <cmath>
#include <stdexcept>

using namespace std;

//------------------------------------------------------- Personal Include
#include "Reward.h"
#include "../Utils.h"

//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
NNRewards::NNRewards(int W, int H)
    : winningMove(2.f * W * H),
      losingMove(-winningMove),
      killingSetupMove(15.f * sqrt(float(W) * H)),
      killZoneEnterMove(6.f * sqrt(float(W) * H)),
      escapeReductionMove(2.4f * sqrt(float(W) * H)),
      survivingMove(0.1f * sqrt(float(W) * H))
// Algorithm : Scale outcome rewards by area and shaping rewards by its square root.
{
}  //----- end of NNRewards


Reward::Reward(const GameConfig &p_config)
    : config(p_config), rewards(config.W, config.H)
// Algorithm : Validate the game rules that determine the reward scales.
{
    validateGameConfig(config);
}  //----- end of Reward


float Reward::computeReward(const GameState &previous, const GameState &next, int player) const
// Algorithm : Score terminal outcomes first, then observed eliminations and continuing play.
{
    if (!(previous.config == config) || !(next.config == config))
    {
        throw invalid_argument("Reward snapshots must match the game configuration");
    }

    const auto &learner = next.players.at(player);

    // Scale outcomes by elapsed scheduled slots, including dead slots, against the legacy turn budget.
    float maxTurns = float(config.W) * config.H * config.M;
    if (config.M > 1)
    {
        maxTurns *= 0.8f;
    }

    float elapsed = float(next.round) * config.N + next.currentPlayer - 1;
    float progress = elapsed / maxTurns;
    float win = rewards.winningMove * (1.f + progress);
    float loss = rewards.losingMove * (0.2f + 0.8f * progress * progress);

    if (!learner.playerInfo.alive)
    {
        return loss;
    }

    int opponentsAlive = 0;
    int opponentsEliminated = 0;
    for (const auto &[opponent, current] : next.players)
    {
        if (opponent != player)
        {
            opponentsAlive += current.playerInfo.alive;
            opponentsEliminated += previous.players.at(opponent).playerInfo.alive &&
                                   !current.playerInfo.alive;
        }
    }

    // A sole survivor wins even if its mask is empty. The win absorbs the final elimination.
    if (opponentsAlive == 0)
    {
        return win;
    }

    if (learner.playerInfo.actionMask.none())
    {
        return loss;
    }

    float score = win * opponentsEliminated + rewards.survivingMove * learner.snake.size();
    for (int opponent = 1; opponent <= config.N; opponent++)
    {
        const auto &current = next.players.at(opponent);

        if (opponent == player || !current.playerInfo.alive)
        {
            continue;
        }

        int previousMoves = previous.players.at(opponent).playerInfo.actionMask.count();
        int nextMoves = current.playerInfo.actionMask.count();

        if (previousMoves > 2 && nextMoves <= 2)
        {
            score += rewards.killingSetupMove;
        }

        float wallProximity = computeKillZoneProximity(current.snake.front(), config);
        // Keep the legacy sigmoid and measure lost escape routes as a fraction of the four directions.
        score += rewards.killZoneEnterMove / (1.f + exp(-4.f * (wallProximity - 0.3f)));
        score += rewards.escapeReductionMove * (max(0, previousMoves - nextMoves) / 4.f);
    }

    return score;
}  //----- end of computeReward
