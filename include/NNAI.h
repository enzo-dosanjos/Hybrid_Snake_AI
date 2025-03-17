/*************************************************************************
NNAI - Full DQN implementation for the game of Snake
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//------------- Interface of the module <NNAI> (file NNAI.h) -------------
#ifndef NNAI_H
#define NNAI_H

//------------------------------------------------------------------------
// Role of the <NNAI> module
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
// Reward configuration  todo: optimise (genetic algo?)
struct NNRewards {
    const float winningMove;
    const float losingMove;

    // Aggression rewards
    const float killingSetupMove;       // When the opps have less than 2 escape moves
    const float killZoneEnterMove;       // When moving an opp into a kill zone (close to corner/walls)
    const float escapeReductionMove;      // Per opponent escape route closed

    // Anti-passivity
    const float entropyPenalty;      // For low action diversity

    const float survivingMove;

    // Constructor takes parent reference
    NNRewards(
        int W,
        int H,
        int turn,
        int maxTurns
    ) :
    winningMove((2.0 * W * H) * (1.0 + turn/float(maxTurns))),  // Reward later wins more
    losingMove(-2.0 * W * H * (0.2 + 0.8*pow(turn/float(maxTurns), 2))), // Curved penalty

    killingSetupMove(15.0 * sqrt(W*H)),
    killZoneEnterMove(6.0 * sqrt(W*H)),
    escapeReductionMove(2.4 * sqrt(W*H)),

    entropyPenalty(-3.0 * sqrt(W*H)),
    survivingMove(0.1 * sqrt(W*H))
    {}
};


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
class NNAI
{
//--------------------------------------------------------- Public Methods
    public:
        void copyWeights (
            const NNAI &src
        );
        // Usage :
        //
        // Contract :
        //

        std::string selectAction(
            const std::vector<float> &boardState,
            const std::vector<float> &extraState
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

        float computeReward(
            const vector<float> &boardState,
            const vector<float> &extraState,
            const vector<float> &prevExtraState,
            const PlayersData &playerState,
            int myPlayer,
            int nbOppKillSetup,
            bool oppWillDie,
            bool oppDied,
            int turn
        );
        // Usage :
        //
        // Contract :
        //

        void trainNN (
            std::vector<Transition> &batch,
            NNAI &targetNetwork,
            float learningRate,
            float gamma
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
        NNAI(
              int w,
              int h,
              int m,
              int n,
              int p,
              int boardC,
              float eps = 1.0,
              float minEps = 0.01,
              float epsDecay = 0.995,
              int replayBufferSize = 100000
          ) : W(w),
              H(h),
              M(m),
              N(n),
              P(p),
              boardChannels(boardC),
              helper(w, h, m, n, p, boardC),
              aiCNN(W * H * boardC, w, h, m, n, p, boardC),
              aiFCNN(-1, w, h, m, n, p, boardC),  // the input size depends on the CNN output
              replayBuffer(replayBufferSize),
              epsilon(eps),
              epsilonMin(minEps),
              epsilonDecay(epsDecay),
              gen(std::random_device{}())
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
            const ConvolutionLayer &layer
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
        int W, H, M, N, P;
        int boardChannels;

        Utils helper;

    public:
        CNN aiCNN;
        FCNN aiFCNN;

        ReplayBuffer replayBuffer;

    private:
        float epsilon;         // Initial exploration rate
        float epsilonMin;      // Minimum value of epsilon
        float epsilonDecay;    // Decay factor for epsilon

        std::mt19937 gen;
};



#endif //NNAI_H
