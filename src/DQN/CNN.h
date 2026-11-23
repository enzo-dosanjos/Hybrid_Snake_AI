/*************************************************************************
CNN - A convolution neural network
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//-------------- Interface of the module <CNN> (file CNN.h) --------------
#ifndef CNN_H
#define CNN_H

//------------------------------------------------------------------------
// Role of the <CNN> module
// Processes variable-size boards using convolution and channel-wise pooling.
//------------------------------------------------------------------------

//--------------------------------------------------------- System Include
#include <random>
#include <string>
#include <vector>

//-------------------------------------------------------- Used interfaces
#include "../Structs.h"
#include "NeuralNetwork.h"

//------------------------------------------------------------------ Types
struct ConvolutionalLayer {
    std::string type;
    int inputChannels;
    int outputChannels;
    int kernelSize;
    int stride;
    int padding;

    // Activations are indexed by [channel][y][x]. The input includes zero padding.
    FeatureTensor input;
    FeatureTensor output;
    FeatureTensor error;

    // Kernels are indexed by [input channel][output channel][kernel y][kernel x].
    Matrix<Matrix<float>> weights;
    std::vector<float> biases;
    Matrix<Matrix<float>> weightGradient;
    std::vector<float> biasGradient;
};


//----------------------------------------------------------------- PUBLIC
class CNN : public NeuralNetwork {
//--------------------------------------------------------- Public Methods
    public:
        void addLayer(const ConvLayerConfig &config);
        // Contract : Infer input channels from the preceding layer or constructor.

        void copyWeights(const CNN &src);
        // Contract : Copy only parameters between networks with matching architectures.

        FeatureTensor forwardPropagation(const BoardTensor &input);
        // Contract : Cache one sample's activations and derive each output shape from its input.

        static std::vector<float> globalAvgPooling(const FeatureTensor &input);
        // Contract : Return the spatial mean of each feature channel.

        static FeatureTensor globalAvgPoolingBackward(const std::vector<float> &pooledGradient,
                                                      const FeatureTensor &input);
        // Contract : Use the shape of the tensor that was pooled to distribute its channel gradients.

        void backPropagation(const FeatureTensor &error);
        // Contract : Backpropagate the gradient with respect to the latest forward output.

        void resetGradients() override;
        void accumulateGradients() override;
        void updateWeights(float learningRate) override;

//--------------------------------------------------------- SETTERS/GETTERS
        int getNNSize() const
        // Algorithm : Return the number of configured layers.
        {
            return nn.size();
        }  //----- end of getNNSize


        std::vector<ConvolutionalLayer> getNN() const
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
        CNN(int inputChannels, const std::mt19937 &p_gen);

//---------------------------------------------------------------- PRIVATE
    private:
        static FeatureTensor padInput(const FeatureTensor &input, int padding);
        static void convolution(ConvolutionalLayer &layer, const FeatureTensor &input);

        std::vector<ConvolutionalLayer> nn;
        int nnInputChannels;
        std::mt19937 gen;
};

#endif //CNN_H
