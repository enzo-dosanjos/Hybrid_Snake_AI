/*************************************************************************
FCNN - A fully connected neural network implementing a Q-learning agent
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//--------- Implementation of the module <FCNN> (file FCNN.cpp) ---------

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <iostream>
#include <algorithm>

using namespace std;

//------------------------------------------------------- Personal Include
#include "../include/FCNN.h"

//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
void FCNN::addLayer (string type, int outputSize, int inputSize)
// Algorithm : Add a fully connected layer to the neural network by
// initialising the type of the activation function, the input and output size
{
    FullyConnectedLayer layer;

    layer.type = type;
    layer.outputSize = outputSize;

    // get the input size either from parameters or from the previous layer's output
    if (inputSize == -1)
    {
        layer.inputSize = (nn.empty()) ? nnInputSize : nn.back().outputSize;
    }
    else
    {
        layer.inputSize = inputSize;
    }

    layer.position = nn.size();

    nn.push_back(layer);
    initLayer(nn.back());
} //----- end of addLayer


void FCNN::copyWeights (const FCNN &src)
// Algorithm : Copy the weights and biases of the fully connected layers
{
    // copy weights and biases of the fully connected layers
    for (int layer = 0; layer < nn.size(); layer++)
    {
        nn[layer].weights = src.nn[layer].weights;
        nn[layer].biases = src.nn[layer].biases;
    }
}  //----- End of copyWeights


vector<float> FCNN::forwardPropagation(const vector<float> &input)
// Algorithm : Compute the output of the neural network by propagating the input
{
    for (auto &layer : nn)
    {
        if (layer.position == 0)
        {
            // The first layer directly receives the input
            computeLayer(layer, input);
        }
        else
        {
            // For the next layers, it uses the previous layer's output
            computeLayer(layer);
        }
    }

    return nn.back().output;
}  //----- End of forwardPropagation


void FCNN::resetGradients ()
// Algorithm : Reset the gradients of every layers
{
    // For each layer other than the input layer, put the gradients back to 0
    for (int layer = 1; layer < nn.size(); layer++)
    {
        for (int i = 0; i < nn[layer].inputSize; i++)
        {
            fill(nn[layer].weightGradient[i].begin(), nn[layer].weightGradient[i].end(), 0.0);
        }

        fill(nn[layer].biasGradient.begin(), nn[layer].biasGradient.end(), 0.0);
    }
}  //----- End of resetGradients


void FCNN::computeError (const vector<float> targetOutput, Transition &transition, float gamma)  // todo: implement n-step q-learning
// Algorithm : Compute the error of the output layer based on the target Q-value,
// the highest value of the target output and the Bellman equation
{
    // If the current episodes has ended, there's no futur so 0
    float maxQ = transition.done ? 0.0 : *max_element(targetOutput.begin(), targetOutput.end());

    // Bellman equation : Compute the target Q-value for the action
    float targetQ = transition.reward + gamma * maxQ;  // gamma controls the importance of future rewards

    // Compute the loss for the taken action, the other are set to 0
    for (int i = 0; i < nn.back().outputSize; i++)
    {
        if (LABEL_ACTIONS[i] == transition.action)
        {
            float error = nn.back().output[i] - targetQ;
            nn.back().error[i] = error;
            transition.tdError = error;
        }
        else
        {
            nn.back().error[i] = 0.0;
        }
    }
}


void FCNN::accumulateGradient()
// Algorithm : Accumulate the weight and bias gradients for the layer
{
    for (auto &layer : nn)
    {
        // Accumulate the weight gradient
        for (int i = 0; i < layer.outputSize; i++)  // For each output neuron
        {
            for (int j = 0; j < layer.inputSize; j++)  // For each input neuron
            {
                layer.weightGradient[j][i] += layer.error[i] * nn[layer.position-1].output[j];
            }

            // Accumulate the bias gradient
            layer.biasGradient[i] += layer.error[i];
        }
    }
} //----- end of accumulateGradient


void FCNN::updateWeights(float learning_rate)
// Algorithm : Update the weights and biases of every layer according to the gradient
{
    for (auto &layer : nn)
    {
        for (int i = 0; i < layer.outputSize; i++)  // For each neuron of the layer
        {
            for (int j = 0; j < layer.inputSize; j++)  // For each input
            {
                layer.weights[j][i] -= learning_rate * layer.weightGradient[j][i];
            }

            // Update the bias
            layer.biases[i] -= learning_rate * layer.biasGradient[i];
        }
    }
} //----- end of updateWeights


void FCNN::gradientDescent(float learning_rate)
// Algorithm : Perform the gradient descent on the neural network
{
    // Accumulate the gradients
    accumulateGradient();

    // Update weights and biases according to the gradient
    updateWeights(learning_rate);

} //----- end of gradientDescent


//-------------------------------------------------------------- PROTECTED
void FCNN::initLayer (FullyConnectedLayer &layer)
// Algorithm : Initialize the weights and biases of the layer
{
    // Resize weights and biases matrices
    layer.weights.resize(layer.inputSize, vector<float>(layer.outputSize));
    layer.biases.resize(layer.outputSize, 0.0);  // initialise bias to 0

    // initialise weight and bias gradients to 0
    layer.weightGradient.resize(layer.inputSize, vector<float>(layer.outputSize, 0.0));
    layer.biasGradient.resize(layer.outputSize, 0.0);

    // initialise weight array with random values
    uniform_real_distribution<float> dist(-0.1, 0.1);  // Uniform distribution between -0.1 and 0.1;
    for (int i = 0; i < layer.inputSize; i++)
    {
        for (int j = 0; j < layer.outputSize; j++)
        {
            layer.weights[i][j] = dist(gen);
        }
    }

    // Resize and initialise error, output
    layer.error.resize(layer.outputSize, 0.0);
    layer.output.resize(layer.outputSize, 0.0);
} //----- end of initLayer


void FCNN::computeLayer (FullyConnectedLayer &layer, const vector<float> &in)
// Algorithm : Compute the output of the layer by matrix multiplication and addition
{
    vector<float> input = (layer.position == 0) ? in : nn[layer.position-1].output;

    // Matrix operation to get the output
    layer.output = multiplyMatrix(input, layer.weights);
    layer.output = addMatrix(layer.output, layer.biases);

    // activation functions
    if (layer.type == "ReLU")
    {
        layer.output = ReLU(layer.output);
    }
} //----- end of computeLayer


void FCNN::backPropagation()
// Algorithm : Compute the error of the layer by backpropagating the error
// from the next layer
{
    for(auto it = nn.rbegin() + 1; it != nn.rend(); it++)
    {
        fill(it->error.begin(), it->error.end(), 0.0);
        for (int i = 0; i < it->outputSize; i++)
        {
            for (int j = 0; j < nn[it->position + 1].outputSize; j++)
            {
                it->error[i] += nn[it->position+1].weights[i][j] * nn[it->position+1].error[j];
            }

            if (it->type == "ReLU")
            {
                // Derivative of ReLU: 1 if x > 0, 0 otherwise
                it->error[i] *= (it->output[i] > 0 ? 1.0 : 0.0);
            }
        }
    }
} //----- end of backPropagation


inline vector<float> FCNN::ReLU(const vector<float> &input)
// Algorithm : Apply the ReLU activation function to the input
{
    vector<float> output = input;
    for (float &x : output)
    {
        if (x < 0) x = 0;
    }

    return output;
} //----- end of ReLU


inline vector<float> FCNN::multiplyMatrix(const vector<float> &row, const Matrix<float> &mat)
// Algorithm : Multiply a row vector by a matrix
{
    const int resM = mat[0].size();
    vector<float> result(resM, 0.0);

    // Optimized multiplication
    for (int i = 0; i < row.size(); i++)
    {
        const double row_val = row[i];
        for (int j = 0; j < resM; j++)
        {
            result[j] += row_val * mat[i][j];
        }
    }

    return result;
} //----- end of multiplyMatrix


inline vector<float> FCNN::addMatrix(const vector<float> &c1, const vector<float> &c2)
// Algorithm : Add two vectors element-wise
{
    const int resM = c2.size();
    vector<float> result(resM, 0.0);

    for (int i = 0; i < resM; i++)
    {
        result[i] = c1[i] + c2[i];
    }

    return result;
} //----- end of addMat