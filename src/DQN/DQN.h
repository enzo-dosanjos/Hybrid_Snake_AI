/*************************************************************************
DQN - Full DQN implementation for the game of Snake
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//------------- Interface of the module <DQN> (file DQN.h) -------------
#ifndef DQN_H
#define DQN_H

//------------------------------------------------------------------------
// Role of the <DQN> module
// This module is a class that implements a full Deep Q-Network (DQN) with
// a Convolutional Neural Network (CNN) and a Fully Connected Neural Network
// (FCNN) to play the game of Snake. It includes methods to train the
// network, select actions, save and load the model, and save and load the
// replay buffer.
//------------------------------------------------------------------------

//-------------------------------------------------------- Used interfaces
#include "Structs.h"
#include "ReplayBuffer.h"
#include "CNN.h"
#include "FCNN.h"

//------------------------------------------------------------------ Types
struct ValidationResult {
    bool victory;
    int survivalTurns;
    float avgReward;
    float maxReward;
    float minReward;
    float actionEntropy;
    std::map<std::string, int> actionCounts;
    float qValueConsistency;
    float deathPenalty;
};

//----------------------------------------------------------------- PUBLIC
class DQN
{
//--------------------------------------------------------- Public Methods
    public:


        std::string selectAction(
            const std::vector<float> &boardState,
            const std::vector<float> &extraState
        );
        // Usage :
        //
        // Contract :
        //

        void copyWeights (
            const DQN &src
        );
        // Usage :
        //
        // Contract :
        //

        std::vector<float> forwardPropagation(
            const std::vector<float> &boardState,
            const std::vector<float> &extraState
        );
        // Usage :
        //
        // Contract :
        //

        std::vector<float> computeError (
            const std::vector<float> &targetOutput,
            Transition &transition
        );
        // Usage :
        //
        // Contract :
        //

        void spreadError(
            const std::vector<float> &fcnnError
        );
        // Usage :
        //
        // Contract :
        //

        void train (
            std::vector<Transition> &batch,
            float learningRate
        );
        // Usage :
        //
        // Contract :
        //

        ValidationResult analyzeResults(
            const std::string &filename
        );
        // Usage :
        //
        // Contract :
        //

        float calculateScore(
            const ValidationResult& result
        );
        // Usage :
        //
        // Contract :
        //

        void validate(
            const std::vector<GameConfig> &validationSet
        );
        // Usage :
        //
        // Contract :
        //

        void saveToFile(
            const std::string &filename,
            int trainingStep
        );
        // Usage :
        //
        // Contract :
        //

        void loadFromFile(
            const std::string &filename,
            int &trainingStep
        );
        // Usage :
        //
        // Contract :
        //

        void saveModelToFile(
            const std::string &filename
        );
        // Usage :
        //
        // Contract :
        //

        void loadModelFromFile(
            const std::string &filename
        );
        // Usage :
        //
        // Contract :
        //

        void saveReplayBufferToFile(
            const std::string &filename,
            int trainingStep
        );
        // Usage :
        //
        // Contract :
        //

        void loadReplayBufferFromFile(
            const std::string &filename,
            int &trainingStep
        );
        // Usage :
        //
        // Contract :
        //

//---------------------------------------------- Constructors - destructor
        DQN(
              const GameConfig &config,
              int boardChannels,
              int replayBufferSize = 100000,
              float p_epsilon = 1.0,
              float p_minEpsilon = 0.01,
              float p_epsilonDecay = 0.995,
              const std::mt19937 &p_gen
          ) : cnn(config.W * config.H * boardChannels, p_gen),
              fcnn(-1, p_gen),  // the input size depends on the CNN output
              replayBuffer(replayBufferSize),
              epsilon(p_epsilon),
              minEpsilon(p_minEpsilon),
              epsilonDecay(p_epsilonDecay),
              gen(p_gen)
        {
            #ifdef MAP
                cout << "Calling the constructor of NNAI" << endl;
            #endif
        }


//-------------------------------------------------------------- PROTECTED
    protected:
        std::vector<float> combineFeatures(
            const std::vector<float> &boardState,
            const std::vector<float> &extras
        );
        // Usage :
        //
        // Contract :
        //

        void decayEpsilon();
        // Usage :
        //
        // Contract :
        //

        void saveLayer(
            std::ofstream &file,
            const FullyConnectedLayer &layer
        );
        // Usage :
        //
        // Contract :
        //

        void saveConvLayer(
            std::ofstream &file,
            const ConvolutionalLayer &layer
        );
        // Usage :
        //
        // Contract :
        //

        void saveVector(
            std::ofstream &file,
            const std::vector<float> &vec
        );
        // Usage :
        //
        // Contract :
        //

        void saveString(
            std::ofstream &file,
            const std::string &str
        );
        // Usage :
        //
        // Contract :
        //

        void loadLayers(
            std::ifstream &file
        );
        // Usage :
        //
        // Contract :
        //

        void loadConvLayers(
            std::ifstream &file
        );
        // Usage :
        //
        // Contract :
        //

        void loadVector(
            std::ifstream &file,
            std::vector<float> &vec
        );
        // Usage :
        //
        // Contract :
        //

        void loadString(
            std::ifstream &file,
            std::string &str
        );
        // Usage :
        //
        // Contract :
        //


//---------------------------------------------------------------- PRIVATE
    private:
        // Neural Networks
        CNN cnn;
        FCNN fcnn;

        // Target Networks
        CNN targetCnn;
        FCNN targetFcnn;

        ReplayBuffer replayBuffer;

    private:
        float epsilon;         // Initial exploration rate
        float minEpsilon;      // Minimum value of epsilon
        float epsilonDecay;    // Decay factor for epsilon
        float gamma;           // Discount factor for future rewards

        std::mt19937 gen;
};



#endif //DQN_H
