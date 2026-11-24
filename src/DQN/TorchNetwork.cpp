/*************************************************************************
TorchNetwork - Batched CPU computation for the optional LibTorch trainer
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//--------- Implementation of the module <TorchNetwork> (file TorchNetwork.cpp) ---------

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <map>
#include <torch/csrc/autograd/autograd.h>

using namespace std;

//------------------------------------------------------- Personal Include
#include "TorchNetwork.h"

//----------------------------------------------------------------- PUBLIC
TorchNetwork::TorchNetwork(const CNN &cnn, const FCNN &fcnn)
// Algorithm : Import seeded or checkpoint parameters from input-major to output-major layout.
{
    for (const auto &layer : cnn.getNN())
    {
        vector<float> values;
        for (const auto &input : layer.weights)
        {
            for (const auto &kernel : input)
            {
                for (const auto &row : kernel)
                {
                    values.insert(values.end(), row.begin(), row.end());
                }
            }
        }

        parameters.push_back(at::tensor(values).reshape({layer.inputChannels, layer.outputChannels,
                             layer.kernelSize, layer.kernelSize}).transpose(0, 1).contiguous());
        parameters.push_back(at::tensor(layer.biases));
    }

    for (const auto &layer : fcnn.getNN())
    {
        vector<float> values;
        for (const auto &row : layer.weights)
        {
            values.insert(values.end(), row.begin(), row.end());
        }
        parameters.push_back(at::tensor(values).reshape({layer.inputSize, layer.outputSize}).t().contiguous());
        parameters.push_back(at::tensor(layer.biases));
    }
}  //----- end of TorchNetwork


void TorchNetwork::setThreads(int count)
// Algorithm : Set the CPU intra-operation thread budget before allocating network tensors.
{
    at::set_num_threads(count);
}  //----- end of setThreads


bool TorchNetwork::empty() const
// Algorithm : Detect a network awaiting initialization from native parameters.
{
    return parameters.empty();
}  //----- end of empty


vector<vector<float>> TorchNetwork::checkpointRows() const
// Algorithm : Flatten input-major weights followed by biases in the existing checkpoint order.
{
    vector<vector<float>> rows;
    for (size_t i = 0; i < parameters.size(); i += 2)
    {
        auto weights = parameters[i].transpose(0, 1).contiguous();
        const auto &bias = parameters[i + 1];
        rows.emplace_back(weights.data_ptr<float>(), weights.data_ptr<float>() + weights.numel());
        rows.back().insert(rows.back().end(), bias.data_ptr<float>(), bias.data_ptr<float>() + bias.numel());
    }
    return rows;
}  //----- end of checkpointRows


array<float, 4> TorchNetwork::predict(const EncodedState &state, const ModelConfig &config) const
// Algorithm : Evaluate one board without recording gradients and return its four action values.
{
    at::NoGradGuard guard;
    auto values = forward({&state}, config);
    array<float, 4> result;
    copy_n(values.data_ptr<float>(), result.size(), result.begin());
    return result;
}  //----- end of predict


void TorchNetwork::update(const vector<Transition> &batch, const TorchNetwork &target,
                          const ModelConfig &config, float rate)
// Algorithm : Differentiate the mean masked Bellman error and publish one SGD update.
{
    vector<const EncodedState *> states, successors;
    vector<int64_t> actions, indices;
    vector<float> rewards, penalties;

    for (const auto &transition : batch)
    {
        if (!transition.done && transition.nextState.legalActions.any())
        {
            indices.push_back(states.size());
            successors.push_back(&transition.nextState);
            for (size_t i = 0; i < ACTIONS.size(); i++)
            {
                penalties.push_back(transition.nextState.legalActions[i] ? 0.f : -numeric_limits<float>::infinity());
            }
        }
        states.push_back(&transition.state);
        actions.push_back(static_cast<int>(transition.action));
        rewards.push_back(transition.reward);
    }

    auto expected = at::tensor(rewards);
    if (!successors.empty())
    {
        auto next = target.forward(successors, config) + at::tensor(penalties).reshape({-1, 4});
        expected.index_add_(0, at::tensor(indices), config.gamma * get<0>(next.max(1)));
    }

    // Separate autograd leaves keep a copied target snapshot independent of the online update.
    TorchNetwork learning = *this;
    for (auto &parameter : learning.parameters)
    {
        parameter = parameter.detach().set_requires_grad(true);
    }
    auto chosen = learning.forward(states, config).gather(1, at::tensor(actions).unsqueeze(1)).flatten();
    auto gradients = torch::autograd::grad({(chosen - expected).square().mean() * 0.5}, learning.parameters);
    for (size_t i = 0; i < parameters.size(); i++)
    {
        parameters[i] = parameters[i].sub(gradients[i], rate);
    }
}  //----- end of update


//---------------------------------------------------------------- PRIVATE
at::Tensor TorchNetwork::forward(const vector<const EncodedState *> &states, const ModelConfig &config) const
// Algorithm : Batch equal board sizes, then restore replay order when shapes differ.
{
    map<pair<int, int>, vector<int64_t>> groups;
    int channels = CHANNELS_PER_PLAYER * config.numPlayers + 1;
    for (size_t i = 0; i < states.size(); i++)
    {
        const auto &board = states[i]->board;
        if (static_cast<int>(board.size()) != channels || board.front().empty() || board.front().front().empty() ||
            static_cast<int>(states[i]->extraFeatures.size()) != config.extraFeatureCount)
        {
            throw invalid_argument("Invalid encoded state shape");
        }
        groups[{board.front().size(), board.front().front().size()}].push_back(i);
    }

    vector<at::Tensor> outputs(states.size());
    for (const auto &[shape, indices] : groups)
    {
        auto [height, width] = shape;
        vector<float> cells, extras;
        cells.reserve(indices.size()*channels*height*width);
        for (auto index : indices)
        {
            const auto &state = *states[index];
            for (const auto &channel : state.board)
            {
                if (static_cast<int>(channel.size()) != height)
                {
                    throw invalid_argument("Inconsistent board height");
                }
                for (const auto &row : channel)
                {
                    if (static_cast<int>(row.size()) != width)
                    {
                        throw invalid_argument("Inconsistent board width");
                    }
                    cells.insert(cells.end(), row.begin(), row.end());
                }
            }
            extras.insert(extras.end(), state.extraFeatures.begin(), state.extraFeatures.end());
        }

        auto board = at::tensor(cells).reshape({static_cast<int64_t>(indices.size()), channels, height, width});
        size_t offset = 0;
        for (const auto &layer : config.convolutionLayers)
        {
            board = at::conv2d(board, parameters[offset], parameters[offset + 1], {layer.stride, layer.stride},
                               {layer.padding, layer.padding});
            if (layer.type == "ReLU")
            {
                board = board.relu();
            }
            offset += 2;
        }

        auto features = at::tensor(extras).reshape({static_cast<int64_t>(indices.size()), config.extraFeatureCount});
        auto values = at::cat({board.mean({2, 3}), features}, 1);
        while (offset < parameters.size())
        {
            values = at::linear(values, parameters[offset], parameters[offset + 1]);
            offset += 2;
            if (offset < parameters.size())
            {
                values = values.relu();
            }
        }
        if (groups.size() == 1)
        {
            return values;
        }
        for (size_t i = 0; i < indices.size(); i++)
        {
            outputs[indices[i]] = values[i];
        }
    }
    return at::stack(outputs);
}  //----- end of forward
