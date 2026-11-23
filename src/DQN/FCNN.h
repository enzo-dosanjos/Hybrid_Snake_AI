/*************************************************************************
FCNN - A fully connected neural network
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//------------- Interface of the module <FCNN> (file FCNN.h) -------------
#ifndef FCNN_H
#define FCNN_H

//------------------------------------------------------------------------
// Role of the <FCNN> module
// Maps pooled board channels and extra features to dense network outputs.
//------------------------------------------------------------------------

//--------------------------------------------------------- System Include
#include <random>
#include <string>
#include <vector>

//-------------------------------------------------------- Used interfaces
#include "../Structs.h"
#include "NeuralNetwork.h"

//------------------------------------------------------------------ Types
struct FullyConnectedLayer {
    std::string type;
    int inputSize;
    int outputSize;

    std::vector<float> input;
    std::vector<float> output;
    std::vector<float> error;

    Matrix<float> weights;
    std::vector<float> biases;
    Matrix<float> weightGradient;
    std::vector<float> biasGradient;
};


//----------------------------------------------------------------- PUBLIC
class FCNN : public NeuralNetwork {
//--------------------------------------------------------- Public Methods
    public:
        void addLayer(const std::string &type, int outputSize);
        // Contract : Infer the input size from the preceding layer or constructor.

        void copyWeights(const FCNN &src);
        // Contract : Copy only parameters between networks with matching architectures.

        std::vector<float> forwardPropagation(const std::vector<float> &input);
        // Contract : Cache one sample's input and layer activations.

        std::vector<float> backPropagation(const std::vector<float> &error);
        // Contract : Backpropagate the gradient with respect to the latest forward output.
        // Return the gradient with respect to the original input.

        void resetGradients() override;
        void accumulateGradients() override;
        void updateWeights(float learningRate) override;

//--------------------------------------------------------- SETTERS/GETTERS
        int getNNSize() const
        // Algorithm : Return the number of configured layers.
        {
            return nn.size();
        }  //----- end of getNNSize


        std::vector<FullyConnectedLayer> getNN() const
        // Algorithm : Return a snapshot of the layers and their training state.
        {
            return nn;
        }  //----- end of getNN


        void clearNN()
        // Algorithm : Remove all configured layers and their cached state.
        {
            nn.clear();
        }  //----- end of clearNN

//---------------------------------------------- Constructors - destructor
        FCNN(int inputSize, const std::mt19937 &p_gen);

//---------------------------------------------------------------- PRIVATE
    private:
        friend class ModelPersistence;

        static void computeLayer(FullyConnectedLayer &layer, const std::vector<float> &input);

        std::vector<FullyConnectedLayer> nn;
        int nnInputSize;
        std::mt19937 gen;
};

#endif //FCNN_H
