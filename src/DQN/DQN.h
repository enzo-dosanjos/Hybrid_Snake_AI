/*************************************************************************
DQN - Evaluates actions and learns from replayed Snake decisions
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//------------- Interface of the module <DQN> (file DQN.h) -------------
#ifndef DQN_H
#define DQN_H

//------------------------------------------------------------------------
// Role of the <DQN> module
// Combines pooled convolutional features with analytical inputs and trains
// an online action-value network against a separate target network.
//------------------------------------------------------------------------

//--------------------------------------------------------- System Include
#include <array>
#include <optional>
#include <random>
#include <vector>

//-------------------------------------------------------- Used interfaces
#include "../Structs.h"
#include "CNN.h"
#include "FCNN.h"
#include "ReplayBuffer.h"

//----------------------------------------------------------------- PUBLIC
class DQN {
//--------------------------------------------------------- Public Methods
    public:
        std::optional<Action> selectAction(const EncodedState &state, bool explore);
        // Contract : Return no action for an empty mask, otherwise an epsilon-greedy legal action.

        std::optional<Action> sampleExplorationAction(const ActionMask &legalActions);
        // Contract : Sample a uniform legal action with probability epsilon; otherwise return no action.

        std::array<float, 4> forwardPropagation(const EncodedState &state);
        // Contract : Return the four raw online Q-values in Action order.

        void remember(const Transition &transition);
        // Contract : Store an owned copy of the executed decision and its outcome.

        void train(int batchSize, float learningRate);
        // Contract : Apply the mean gradient of a sampled batch and decay epsilon.
        // An empty batch leaves both the parameters and epsilon unchanged.

        void updateTargetNetworks();
        // Contract : Refresh both target networks with the current online parameters.

//--------------------------------------------------------- SETTERS/GETTERS
        const FeatureSelection &getFeatureSelection() const;
        // Usage : Encode decisions using the feature groups configured for this learner.

//---------------------------------------------- Constructors - destructor
        DQN(const ModelConfig &config, const std::mt19937 &p_gen);
        // Contract : Build matching online and target networks for the configured input and layer shapes.

//---------------------------------------------------------------- PRIVATE
    private:
        static CNN makeCNN(const ModelConfig &config, const std::mt19937 &gen);
        static FCNN makeFCNN(const ModelConfig &config, const std::mt19937 &gen);
        static std::array<float, 4> evaluate(FCNN &network, const FeatureTensor &cnnOutput,
                                            const std::vector<float> &extras);
        static int bestLegalIndex(const std::array<float, 4> &values, const ActionMask &legalActions);
        float bellmanTarget(const Transition &transition);
        void decayEpsilon();

        CNN cnn;
        FCNN fcnn;
        CNN targetCNN;
        FCNN targetFCNN;
        ReplayBuffer replayBuffer;
        FeatureSelection featureSelection;
        float epsilon;
        float minEpsilon;
        float epsilonDecay;
        float gamma;
        std::mt19937 gen;
};

#endif //DQN_H
