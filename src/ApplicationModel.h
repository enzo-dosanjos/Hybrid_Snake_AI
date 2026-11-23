/*************************************************************************
ApplicationModel - Shares model and policy construction across executables
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//--- Interface of the module <ApplicationModel> (file ApplicationModel.h) ---
#ifndef APPLICATIONMODEL_H
#define APPLICATIONMODEL_H

//------------------------------------------------------------------------
// Role of the <ApplicationModel> module
// Use compatible network shapes and agent settings in training and play.
//------------------------------------------------------------------------

//--------------------------------------------------------- System Include
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

//-------------------------------------------------------- Used interfaces
#include "Agent/DQNAgent.h"
#include "Agent/HybridAgent.h"
#include "GameEngine/StateAnalyzer.h"

//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
inline ModelConfig makeApplicationModelConfig(int numPlayers)
// Algorithm : Use the default architecture with the encoded feature count for the given number of seats.
{
    ModelConfig config;
    config.numPlayers = numPlayers;
    config.extraFeatureCount = StateAnalyzer::getFeatureCount(numPlayers, config.features);
    return config;
}  //----- end of makeApplicationModelConfig


inline float parseHybridWeight(const std::string &text)
// Algorithm : Read a complete numeric argument within the supported hybrid weight range.
{
    std::size_t end = 0;
    float weight = std::stof(text, &end);
    if (end != text.size() || !(weight >= 0.f && weight <= 1.f))
    {
        throw std::invalid_argument("Hybrid weight must be a number in [0, 1]");
    }

    return weight;
}  //----- end of parseHybridWeight


inline std::unique_ptr<Agent> makeNeuralAgent(DQN &learner, std::optional<float> hybridWeight,
                                             bool explore, const MinimaxParams &params)
// Algorithm : Construct the selected behavior with the shared learner and requested exploration mode.
{
    if (hybridWeight)
    {
        return std::make_unique<HybridAgent>(learner, params, *hybridWeight, explore);
    }

    return std::make_unique<DQNAgent>(learner, explore);
}  //----- end of makeNeuralAgent

#endif //APPLICATIONMODEL_H
