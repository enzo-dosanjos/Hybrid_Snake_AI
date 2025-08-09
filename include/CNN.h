/*************************************************************************
CNN - A convolution neural network implementing a Q-learning agent
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//-------------- Interface of the module <CNN> (file CNN.h) --------------
#ifndef CNN_H
#define CNN_H

//------------------------------------------------------------------------
// Role of the <CNN> module
// the CNN module implements a convolutional neural network (CNN) that is
// used to implement a Q-learning agent. The CNN is used to process the
// board state that can have variable size.
//------------------------------------------------------------------------

//---------------------------------------------------------------  INCLUDE
#include <random>

//-------------------------------------------------------- Used interfaces
#include "Structs.h"
#include "Utils.h"

//------------------------------------------------------------------ Types
struct ConvolutionalLayer {
    std::string type;
    int inputChannels;
    int outputChannels;

    int kernelSize;

    int stride;
    int padding;

    std::pair<int, int> outputSize;

    Matrix<std::vector<float>> input;

    // 4D weights: [inputChannels][outputChannels][kernelSize][kernelSize]
    Matrix<Matrix<float>> weights;
    std::vector<float> biases;
    Matrix<std::vector<float>> output;

    Matrix<Matrix<float>> weightGradient;
    std::vector<float> biasGradient;
    Matrix<std::vector<float>> error;

    int position;
};


//----------------------------------------------------------------- PUBLIC
class CNN
{
//--------------------------------------------------------- Public Methods
    public:
        void addLayer (
            std::string type,
            int kernelSize,
            int stride,
            int padding,
            int outputC,
            std::pair<int, int> outputSize,
            int inputC = -1
        );
        // Usage :
        //
        // Contract :
        //

        void copyWeights(
            const CNN &src
        );
        // Usage :
        //
        // Contract :
        //

        Matrix<std::vector<float>> forwardPropagation(
            const Matrix<std::vector<float>> &input
        );
        // Usage :
        //
        // Contract :
        //

        static std::vector<float> globalAvgPooling(
            const Matrix<std::vector<float>> &input
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

        void spreadFCNNError(
            const std::vector<float> &fcnnError
        );
        // Usage :
        //
        // Contract :
        //

        void backPropagation(
        );
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
    int getNNSize() const
    {
        return cnn.size();
    }
    // Usage :
    //
    // Contract :
    //

    std::vector<ConvolutionalLayer> getNN() const
    {
        return cnn;
    }
    // Usage :
    //
    // Contract :
    //

    void clearNN()
    {
        cnn.clear();
    }


//---------------------------------------------- Constructors - destructor
        CNN(
              int inputSize,
              int boardC
          ) : nnInputSize(inputSize),
              boardChannels(boardC),
              gen(std::random_device{}())
        {
            #ifdef MAP
                cout << "Calling the constructor of cnn" << endl;
            #endif
        }

//-------------------------------------------------------------- PROTECTED
    protected:
        void initLayer(
            ConvolutionalLayer &layer
        );
        // Usage :
        //
        // Contract :
        //

        static Matrix<std::vector<float>> padInput(
            const Matrix<std::vector<float>> &input,
            ConvolutionalLayer &layer
        );

        void convolution (
            ConvolutionalLayer &layer,
            const Matrix<std::vector<float>> &in = Matrix<std::vector<float>>()
        );
        // Usage :
        //
        // Contract :
        //

        float ReLU(
            const float &input
        );
        // Usage :
        //
        // Contract :
        //

//---------------------------------------------------------------- PRIVATE
    private:
        int boardChannels;

        std::vector<ConvolutionalLayer> cnn;
        int nnInputSize;

        std::mt19937 gen;
};



#endif //CNN_H
