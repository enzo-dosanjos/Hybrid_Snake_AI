/*************************************************************************
DQN - Evaluates actions and learns from replayed Snake decisions
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//--------- Implementation of the module <DQN> (file DQN.cpp) ---------

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

using namespace std;

//------------------------------------------------------- Personal Include
#include "DQN.h"

//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
DQN::DQN(const ModelConfig &config, const mt19937 &p_gen)
    : cnn(makeCNN(config, p_gen)), fcnn(makeFCNN(config, p_gen)),
      targetCNN(cnn), targetFCNN(fcnn), replayBuffer(config.replayCapacity, p_gen),
      featureSelection(config.features),
      epsilon(config.epsilon), minEpsilon(config.minEpsilon), epsilonDecay(config.epsilonDecay),
      gamma(config.gamma), gen(p_gen)
// Algorithm : Configure the online networks and copy their initial parameters into the targets.
{
    for (float value : {epsilon, minEpsilon, epsilonDecay, gamma})
    {
        if (!(value >= 0.f && value <= 1.f))
        {
            throw invalid_argument("DQN exploration and discount parameters must be in [0, 1]");
        }
    }
}  //----- end of DQN


optional<Action> DQN::selectAction(const EncodedState &state, bool explore)
// Algorithm : Sample uniformly from legal actions with probability epsilon, otherwise maximize the legal Q-values.
{
    if (state.legalActions.none())
    {
        return nullopt;
    }

    if (explore)
    {
        auto action = sampleExplorationAction(state.legalActions);
        if (action)
        {
            return action;
        }
    }

    return static_cast<Action>(bestLegalIndex(forwardPropagation(state), state.legalActions));
}  //----- end of selectAction


optional<Action> DQN::sampleExplorationAction(const ActionMask &legalActions)
// Algorithm : Use the current epsilon to decide whether to sample uniformly from the supplied mask.
{
    if (legalActions.none() || uniform_real_distribution<float>(0.f, 1.f)(gen) >= epsilon)
    {
        return nullopt;
    }

    array<Action, 4> actions;
    int count = 0;
    for (size_t index = 0; index < legalActions.size(); index++)
    {
        if (legalActions[index])
        {
            actions[count] = static_cast<Action>(index);
            count++;
        }
    }

    return actions[uniform_int_distribution<int>(0, count - 1)(gen)];
}  //----- end of sampleExplorationAction


array<float, 4> DQN::forwardPropagation(const EncodedState &state)
// Algorithm : Encode the board with the online CNN and evaluate it together with the selected extras.
{
    return evaluate(fcnn, cnn.forwardPropagation(state.board), state.extraFeatures);
}  //----- end of forwardPropagation


void DQN::remember(const Transition &transition)
// Algorithm : Check the executed action index before copying the decision into replay.
{
    int action = static_cast<int>(transition.action);
    if (action < 0 || action >= static_cast<int>(ACTIONS.size()))
    {
        throw invalid_argument("Invalid replay action");
    }

    replayBuffer.push(transition);
}  //----- end of remember


void DQN::train(int batchSize, float learningRate)
// Algorithm : Accumulate online gradients for decision-to-decision Bellman targets and apply their mean.
{
    if (!isfinite(learningRate) || learningRate < 0.f)
    {
        throw invalid_argument("DQN learning rate must be finite and nonnegative");
    }

    auto batch = replayBuffer.createBatch(batchSize);
    if (batch.empty())
    {
        return;
    }

    cnn.resetGradients();
    fcnn.resetGradients();

    for (const Transition &transition : batch)
    {
        float target = bellmanTarget(transition);
        auto cnnOutput = cnn.forwardPropagation(transition.state.board);
        auto values = evaluate(fcnn, cnnOutput, transition.state.extraFeatures);
        int action = static_cast<int>(transition.action);

        // The derivative of half-squared TD error is zero for the other action outputs.
        vector<float> outputGradient(values.size(), 0.f);
        outputGradient[action] = values[action] - target;
        auto inputGradient = fcnn.backPropagation(outputGradient);

        // Only the prefix of the dense input came from the CNN.
        inputGradient.resize(cnnOutput.size());
        cnn.backPropagation(CNN::globalAvgPoolingBackward(inputGradient, cnnOutput));

        fcnn.accumulateGradients();
        cnn.accumulateGradients();
    }

    float stepSize = learningRate / batch.size();
    fcnn.updateWeights(stepSize);
    cnn.updateWeights(stepSize);
    decayEpsilon();
}  //----- end of train


void DQN::updateTargetNetworks()
// Algorithm : Copy the latest online weights and biases without changing the online training state.
{
    targetCNN.copyWeights(cnn);
    targetFCNN.copyWeights(fcnn);
}  //----- end of updateTargetNetworks


//--------------------------------------------------------- SETTERS/GETTERS
const FeatureSelection &DQN::getFeatureSelection() const
// Algorithm : Return the feature selection copied at model construction.
{
    return featureSelection;
}  //----- end of getFeatureSelection


//---------------------------------------------------------------- PRIVATE
CNN DQN::makeCNN(const ModelConfig &config, const mt19937 &gen)
// Algorithm : Derive the input channels from the observation schema and add the configured convolution layers.
{
    if (config.observationSchemaVersion != OBSERVATION_SCHEMA_VERSION ||
        config.numPlayers <= 0 ||
        config.numPlayers > (numeric_limits<int>::max() - 1) / static_cast<int>(CHANNELS_PER_PLAYER) ||
        config.convolutionLayers.empty())
    {
        throw invalid_argument("Invalid DQN observation or convolution configuration");
    }

    CNN network(CHANNELS_PER_PLAYER * config.numPlayers + 1, gen);
    for (const auto &layer : config.convolutionLayers)
    {
        network.addLayer(layer);
    }

    return network;
}  //----- end of makeCNN


FCNN DQN::makeFCNN(const ModelConfig &config, const mt19937 &gen)
// Algorithm : Size the dense input from pooled channels and selected extras, then add a linear action head.
{
    int pooledChannels = config.convolutionLayers.back().outputChannels;
    if (config.featureSchemaVersion != EXTRA_FEATURE_SCHEMA_VERSION ||
        config.extraFeatureCount < 0 || config.extraFeatureCount > numeric_limits<int>::max() - pooledChannels)
    {
        throw invalid_argument("Invalid DQN extra-feature configuration");
    }

    FCNN network(pooledChannels + config.extraFeatureCount, gen);
    for (int hiddenSize : config.denseHiddenSizes)
    {
        network.addLayer("ReLU", hiddenSize);
    }

    network.addLayer("linear", ACTIONS.size());
    return network;
}  //----- end of makeFCNN


array<float, 4> DQN::evaluate(FCNN &network, const FeatureTensor &cnnOutput,
                             const vector<float> &extras)
// Algorithm : Concatenate the spatial channel means followed by extras in their encoded order.
{
    auto input = CNN::globalAvgPooling(cnnOutput);
    input.insert(input.end(), extras.begin(), extras.end());
    auto output = network.forwardPropagation(input);

    array<float, 4> values;
    copy(output.begin(), output.end(), values.begin());
    return values;
}  //----- end of evaluate


int DQN::bestLegalIndex(const array<float, 4> &values, const ActionMask &legalActions)
// Algorithm : Find the greatest legal action value, breaking ties by Action order.
{
    int best = -1;
    for (size_t index = 0; index < values.size(); index++)
    {
        if (legalActions[index] && (best < 0 || values[index] > values[best]))
        {
            best = index;
        }
    }

    return best;
}  //----- end of bestLegalIndex


float DQN::bellmanTarget(const Transition &transition)
// Algorithm : Bootstrap one decision ahead from the target network only when a legal next action exists.
{
    const auto &next = transition.nextState;
    if (transition.done || next.legalActions.none())
    {
        return transition.reward;
    }

    auto values = evaluate(targetFCNN, targetCNN.forwardPropagation(next.board), next.extraFeatures);
    return transition.reward + gamma * values[bestLegalIndex(values, next.legalActions)];
}  //----- end of bellmanTarget


void DQN::decayEpsilon()
// Algorithm : Reduce the exploration rate while respecting its configured floor.
{
    if (epsilon > minEpsilon)
    {
        epsilon = max(minEpsilon, epsilon * epsilonDecay);
    }
}  //----- end of decayEpsilon
