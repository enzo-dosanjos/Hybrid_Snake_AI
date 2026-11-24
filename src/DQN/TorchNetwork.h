/*************************************************************************
TorchNetwork - Batched CPU computation for the optional LibTorch trainer
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//--------- Interface of the module <TorchNetwork> (file TorchNetwork.h) ---------
#ifndef TORCHNETWORK_H
#define TORCHNETWORK_H

//------------------------------------------------------------------------
// Role of the <TorchNetwork> module
// Own CPU parameters in pairs of weights and biases for each configured layer.
//------------------------------------------------------------------------
#include <ATen/core/Tensor.h>
#include <array>
#include "CNN.h"
#include "FCNN.h"

class TorchNetwork {
    public:
        TorchNetwork() = default;
        TorchNetwork(const CNN &cnn, const FCNN &fcnn);
        static void setThreads(int count);
        bool empty() const;
        std::vector<std::vector<float>> checkpointRows() const;
        std::array<float, 4> predict(const EncodedState &state, const ModelConfig &config) const;
        void update(const std::vector<Transition> &batch, const TorchNetwork &target,
                    const ModelConfig &config, float rate);

    private:
        at::Tensor forward(const std::vector<const EncodedState *> &states,
                           const ModelConfig &config) const;
        std::vector<at::Tensor> parameters;
};

#endif //TORCHNETWORK_H
