/*************************************************************************
CNN - A convolution neural network
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//--------- Implementation of the module <CNN> (file CNN.cpp) ---------

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <iostream>

using namespace std;

//------------------------------------------------------- Personal Include
#include "CNN.h"

//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
void CNN::addLayer (string type, int kernelSize, int stride, const int padding, int outputC, pair<int, int> outputSize,  int inputC)
// Algorithm : Add a convolutional layer to the neural network by
// initialising the type of the activation function, the input and output size
{
    ConvolutionalLayer layer;

    layer.type = type;
    layer.kernelSize = kernelSize;
    layer.stride = stride;
    layer.padding = padding;
    layer.outputChannels = outputC;
    layer.outputSize = outputSize;

    // get the input size either from parameters or from the previous layer's output
    if (inputC == -1)
    {
        layer.inputChannels = (nn.empty()) ? nnInputSize : nn.back().outputChannels;
    }
    else
    {
        layer.inputChannels = inputC;
    }

    layer.position = nn.size();

    nn.push_back(layer);
    initLayer(nn.back());
}  //----- end of addLayer


void CNN::copyWeights (const CNN &src)
// Algorithm : Copy the weights and biases of the convolutional layers
{
    // copy weights and biases of the convolutional layers
    for (int layer = 0; layer < nn.size(); layer++)
    {
        nn[layer].weights = src.nn[layer].weights;
        nn[layer].biases = src.nn[layer].biases;
    }
}  //----- End of copyWeights


Matrix<vector<float>> CNN::forwardPropagation(const Matrix<vector<float>> &input)
// Algorithm : Compute the output of the neural network by propagating the input
{
    for (auto &layer : nn)
    {
        if (layer.position == 0)
        {
            // The first layer directly receives the input
            convolution(layer, input);
        }
        else
        {

            // For the next layers, it uses the previous layer's output
            convolution(layer);
        }
    }

    return nn.back().output;
}  //----- End of forwardPropagation


vector<float> CNN::globalAvgPooling(const Matrix<vector<float>> &input)
// Algorithm : Compute the global average pooling of the input
{
    vector<float> output(input.size(), 0.0);

    // for each channel
    for(int c = 0; c < input.size(); c++)
    {
        float sum = 0.0;

        // for each cell
        for(auto &row : input[c])
        {
            for(auto &val : row)
            {
                // sum the values
                sum += val;
            }
        }

        // average the sum
        output[c] = sum / (input[c].size() * input[c][0].size());
    }

    return output;
}  //----- End of globalAvgPooling


void CNN::resetGradients ()
// Algorithm : Reset the gradients of every layers
{
    // For each layer other than the input layer, put the gradients back to 0
    for (int layer = 0; layer < nn.size(); layer++)
    {
        for (int i = 0; i < nn[layer].inputChannels; i++)
        {
            for (int j = 0; j < nn[layer].outputChannels; j++)
            {
                for (int k = 0; k < nn[layer].kernelSize; k++)
                {
                    fill(nn[layer].weightGradient[i][j][k].begin(), nn[layer].weightGradient[i][j][k].end(), 0.0);
                }
            }
        }

        fill(nn[layer].biasGradient.begin(), nn[layer].biasGradient.end(), 0.0);
    }
}  //----- End of resetGradients


void CNN::backPropagation(const Matrix<vector<float>> &error)
// Algorithm : Compute the error of every layer by backpropagating the error
// from the last layer
{
    // Set the error of the last layer
    nn.back().error = error;

    for (auto it = nn.rbegin() + 1; it != nn.rend(); ++it)
    {
        ConvolutionalLayer &current_layer = *it;
        ConvolutionalLayer &next_layer = nn[it->position + 1];

        // Reset current layer's error to zero
        for (int oc = 0; oc < current_layer.outputChannels; oc++)
        {
            for (int h = 0; h < current_layer.outputSize.first; h++)
            {
                fill(current_layer.error[oc][h].begin(), current_layer.error[oc][h].end(), 0.0);
            }
        }

        // Iterate over next layer's error positions
        for (int nextoc = 0; nextoc < next_layer.outputChannels; nextoc++)
        {
            for (int ol1_h = 0; ol1_h < next_layer.outputSize.first; ol1_h++)
            {
                for (int ol1_w = 0; ol1_w < next_layer.outputSize.second; ol1_w++)
                {
                    float layerError = next_layer.error[nextoc][ol1_h][ol1_w];

                    // Iterate over kernel positions
                    for (int kh = 0; kh < next_layer.kernelSize; kh++)
                    {
                        for (int kw = 0; kw < next_layer.kernelSize; kw++)
                        {
                            // Calculate corresponding position in current layer's output
                            int ol_h = ol1_h * next_layer.stride - next_layer.padding + kh;
                            int ol_w = ol1_w * next_layer.stride - next_layer.padding + kw;

                            if (ol_h >= 0 && ol_h < current_layer.outputSize.first &&
                                ol_w >= 0 && ol_w < current_layer.outputSize.second)
                            {

                                // Accumulate error for each input channel of the current layer
                                for (int ic = 0; ic < current_layer.outputChannels; ic++)
                                {
                                    current_layer.error[ic][ol_h][ol_w] +=
                                        layerError * next_layer.weights[ic][nextoc][kh][kw];
                                }
                            }
                        }
                    }
                }
            }
        }

        // Apply ReLU derivative
        if (current_layer.type == "ReLU")
        {
            for (int oc = 0; oc < current_layer.outputChannels; oc++)
            {
                for (int h = 0; h < current_layer.outputSize.first; h++)
                {
                    for (int w = 0; w < current_layer.outputSize.second; w++)
                    {
                        current_layer.error[oc][h][w] *= (current_layer.output[oc][h][w] > 0 ? 1.0 : 0.0);
                    }
                }
            }
        }
    }
} //----- end of backPropagation


void CNN::accumulateGradient()
// Algorithm : Accumulate the weight and bias gradients for every layer
{
    for (auto &layer : nn)
    {
        // For each output channel
        for (int oc = 0; oc < layer.outputChannels; oc++)
        {
            // For each output cell
            for (int oh = 0; oh < layer.outputSize.first; oh++)
            {
                for (int ow = 0; ow < layer.outputSize.second; ow++)
                {

                    // For each input channel
                    for (int ic = 0; ic < layer.inputChannels; ic++)
                    {
                        // For each kernel cell
                        for (int kh = 0; kh < layer.kernelSize; kh++)
                        {
                            for (int kw = 0; kw < layer.kernelSize; kw++)
                            {
                                int ih = oh * layer.stride + kh;
                                int iw = ow * layer.stride + kw;

                                if (ih >= 0 && ih < layer.input[ic].size() && iw >= 0 && iw < layer.input[ic][0].size())
                                {
                                    // Weight gradient
                                    layer.weightGradient[ic][oc][kh][kw] += layer.input[ic][ih][iw] * layer.error[oc][oh][ow];
                                }
                            }
                        }
                    }

                    // Accumulate bias gradient
                    layer.biasGradient[oc] += layer.error[oc][oh][ow];
                }
            }
        }
    }
} //----- end of accumulateGradient


void CNN::updateWeights(float learning_rate)
// Algorithm : Update the weights and biases of every layer according to the gradient
{
    for (auto &layer : nn)
    {
        // Update weights
        for (int oc = 0; oc < layer.outputChannels; oc++)
        {
            for (int ic = 0; ic < layer.inputChannels; ic++)
            {
                for (int kh = 0; kh < layer.kernelSize; kh++)
                {
                    for (int kw = 0; kw < layer.kernelSize; kw++)
                    {
                        layer.weights[ic][oc][kh][kw] -= learning_rate
                                                       * layer.weightGradient[ic][oc][kh][kw];
                    }
                }
            }

            // Update biases
            layer.biases[oc] -= learning_rate * layer.biasGradient[oc];
        }
    }
} //----- end of updateWeights

//-------------------------------------------------------------- PROTECTED
void CNN::initLayer (ConvolutionalLayer &layer)
// Algorithm : Initialize the weights and biases of the layer
{
    // Resize weights and biases matrices
    layer.weights.resize(layer.inputChannels);
    layer.biases.resize(layer.outputChannels, 0.0);  // initialise bias to 0

    // initialise weight and bias gradients to 0
    layer.weightGradient.resize(layer.inputChannels);
    layer.biasGradient.resize(layer.outputChannels, 0.0);


    // initialise weight array with random values
    uniform_real_distribution<float> dist(-0.1, 0.1);  // Uniform distribution between -0.1 and 0.1;
    for (int i = 0; i < layer.inputChannels; i++)
    {
        layer.weights[i].resize(layer.outputChannels,
                              Matrix<float>(layer.kernelSize, layer.kernelSize, 0.0));

        layer.weightGradient[i].resize(layer.outputChannels,
                                       Matrix<float>(layer.kernelSize, layer.kernelSize, 0.0));

        for (int j = 0; j < layer.outputChannels; j++)
        {
            for (int k = 0; k < layer.kernelSize; k++)
            {
                for (int l = 0; l < layer.kernelSize; l++)
                {
                    layer.weights[i][j][k][l] = dist(gen);
                }
            }
        }
    }

    // Resize and initialise error, output
    layer.error.resize(layer.outputChannels,
                       Matrix<float>(layer.outputSize.first, layer.outputSize.second, 0.0));
    layer.output.resize(layer.outputChannels,
                        Matrix<float>(layer.outputSize.first, layer.outputSize.second, 0.0));

} //----- end of initLayer


Matrix<vector<float>> CNN::padInput(const Matrix<vector<float>> &input, ConvolutionalLayer &layer)
// Algorithm : Add padding to the input
{
    int inputH = input.empty() ? 0 : input[0].size();
    int inputW = (inputH == 0) ? 0 : input[0][0].size();

    Matrix<vector<float>> paddedInput(input.size(), inputH + 2*layer.padding, vector<float>(inputW + 2*layer.padding, 0.0f));

    for(int c = 0; c < input.size(); c++)
    {
        for(int h = 0; h < inputH; h++)
        {
            for(int w = 0; w < inputW; w++)
            {
                paddedInput[c][h+layer.padding][w+layer.padding] = input[c][h][w];
            }
        }
    }

    return paddedInput;
}


void CNN::convolution (ConvolutionalLayer &layer, const Matrix<vector<float>> &in)
// Algorithm : Compute the output of the layer by convolution
{
    Matrix<vector<float>> input = (layer.position == 0) ? in : nn[layer.position-1].output;

    Matrix<vector<float>> paddedInput = padInput(input, layer);

    layer.input = paddedInput;

    // Convolution
    for(int oc = 0; oc < layer.outputChannels; oc++)  // canals
    {
        for(int oh = 0; oh < layer.outputSize.first; oh++)  // board H
        {
            for(int ow = 0; ow < layer.outputSize.second; ow++)  // board W
            {
                float sum = 0.0;
                for(int ic = 0; ic < layer.inputChannels; ic++)
                {
                    for(int kh = 0; kh < layer.kernelSize; kh++)  // kernel height
                    {
                        for(int kw = 0; kw < layer.kernelSize; kw++)  // kernel width
                        {
                            int ih = oh*layer.stride + kh;
                            int iw = ow*layer.stride + kw;
                            sum += paddedInput[ic][ih][iw] * layer.weights[ic][oc][kh][kw];
                        }
                    }
                }

                // activation functions
                if (layer.type == "ReLU")
                {
                    layer.output[oc][oh][ow] = ReLU(sum + layer.biases[oc]);
                }
            }
        }
    }
} //----- end of computeLayer


float CNN::ReLU(const float &input)
// Algorithm : Apply the ReLU activation function to the input
{
    float output = input;
    if (output < 0)
    {
      output = 0;
    }

    return output;
} //----- end of ReLU
