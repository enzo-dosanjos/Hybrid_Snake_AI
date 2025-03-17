/*************************************************************************
FCNN - A fully connected neural network implementing a Q-learning agent
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//------------- Interface of the module <FCNN> (file FCNN.h) -------------
#ifndef FCNN_H
#define FCNN_H

//------------------------------------------------------------------------
// Role of the <FCNN> module
// The FCNN module implements a fully connected neural network (FCNN) that
// is used to implement a Q-learning agent. The FCNN is used to approximate
// the Q-values of the state-action pairs and is trained using a Deep
// Q-Learning algorithm. The neural network is used to approximate the Q-values.
//------------------------------------------------------------------------

//-------------------------------------------------------- Used interfaces
#include "Structs.h"
#include "ReplayBuffer.h"
#include "Utils.h"

//-------------------------------------------------------------- Constants
std::map<int, std::string> LABEL_ACTIONS = {
    {0, "left"},
    {1, "right"},
    {2, "up"},
    {3, "down"}
};

//------------------------------------------------------------------ Types
struct FullyConnectedLayer {
    std::string type;                // activation function (ex: ReLU)
    int inputSize;
    int outputSize;

    Matrix<float> weights;
    std::vector<float> biases;
    std::vector<float> output;

    Matrix<float> weightGradient;
    std::vector<float> biasGradient;
    std::vector<float> error;

    int position;
};


//----------------------------------------------------------------- PUBLIC
class FCNN
{
//--------------------------------------------------------- Public Methods
    public:
        void addLayer (
            std::string type,
            int outputSize,
            int inputSize = -1
        );
        // Usage :
        //
        // Contract :
        //

        void copyWeights (
            const FCNN &src
        );
        // Usage :
        //
        // Contract :
        //

        std::vector<float> forwardPropagation(
            const std::vector<float> &input
        );
        // Usage :
        //
        // Contract :
        //

        void resetGradients ();
        // Usage :
        //
        // Contract :
        //

        void computeError (
            const std::vector<float> targetOutput,
            Transition &transition, float gamma
        );
        // Usage :
        //
        // Contract :
        //

        void backPropagation();
        // Usage :
        //
        // Contract :
        //

        void accumulateGradient();
        // Usage :
        //
        // Contract :
        //

        void updateWeights(
            float learning_rate
        );
        // Usage :
        //
        // Contract :
        //

        void gradientDescent(
            float learning_rate
        );
        // Usage :
        //
        // Contract :
        //
        

//--------------------------------------------------------- SETTERS/GETTERS
        std::vector<float> getError(
            int layer
        ) const
        {
            return nn[layer].error;
        }
        // Usage :
        //
        // Contract :
        //

        int getNNSize() const
        {
            return nn.size();
        }
        // Usage :
        //
        // Contract :
        //

        std::vector<FullyConnectedLayer> getNN() const
        {
            return nn;
        }
        // Usage :
        //
        // Contract :
        //

        void clearNN()
        {
            nn.clear();
        }

//---------------------------------------------- Constructors - destructor
        FCNN (
            int inputSize,
            int w,
            int h,
            int m,
            int n,
            int p,
            int boardC
        ) : nnInputSize(inputSize),
            W(w),
            H(h),
            M(m),
            N(n),
            P(p),
            boardChannels(boardC),
            helper(w, h, m, n, p, boardC),
            gen(std::random_device()())  // Initialize with a random seed
        {
            #ifdef MAP
                        cout << "Calling the constructor of fcnn" << endl;
            #endif
        }

//-------------------------------------------------------------- PROTECTED
    protected:
        void initLayer (
            FullyConnectedLayer &layer
        );
        // Usage :
        //
        // Contract :
        //

        void computeLayer (
            FullyConnectedLayer &layer,
            const std::vector<float> &in = std::vector<float>()
        );
        // Usage :
        //
        // Contract :
        //

        std::vector<float> ReLU(
            const std::vector<float> &input
        );
        // Usage :
        //
        // Contract :
        //

        std::vector<float> multiplyMatrix(
            const std::vector<float> &row,
            const Matrix<float> &mat
        );
        // Usage :
        //
        // Contract :
        //

        std::vector<float> addMatrix(
            const std::vector<float> &row,
            const std::vector<float> &column
        );
        // Usage :
        //
        // Contract :
        //

//---------------------------------------------------------------- PRIVATE
    private:
        int W, H, M, N, P;
        int boardChannels;

        // NN Components
        std::vector<FullyConnectedLayer> nn;
        int nnInputSize;

        Utils helper;

        // random number generator for layer initialization
        std::mt19937 gen;
};



#endif //FCNN_H
