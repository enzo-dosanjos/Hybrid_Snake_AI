/*************************************************************************
FCNN - A fully connected neural network
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//--------- Implementation of the module <FCNN> (file FCNN.cpp) ---------

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <algorithm>
#include <stdexcept>
#include <utility>

using namespace std;

//------------------------------------------------------- Personal Include
#include "FCNN.h"

//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
FCNN::FCNN(int inputSize, const mt19937 &p_gen) : nnInputSize(inputSize), gen(p_gen)
// Algorithm : Store the size and random generator used to initialize the dense layers.
{
    if (inputSize <= 0)
    {
        throw invalid_argument("FCNN input size must be positive");
    }
}  //----- end of FCNN


void FCNN::addLayer(const string &type, int outputSize)
// Algorithm : Initialize a dense layer sized from the input or preceding layer.
{
    if ((type != "ReLU" && type != "linear") || outputSize <= 0)
    {
        throw invalid_argument("Invalid dense layer configuration");
    }

    FullyConnectedLayer layer;
    layer.type = type;
    layer.inputSize = nn.empty() ? nnInputSize : nn.back().outputSize;
    layer.outputSize = outputSize;
    layer.weights = Matrix<float>(layer.outputSize, layer.inputSize);
    layer.biases.resize(layer.outputSize);
    layer.weightGradient = layer.weights;
    layer.biasGradient.resize(layer.outputSize);

    uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (auto &row : layer.weights)
    {
        for (float &weight : row)
        {
            weight = dist(gen);
        }
    }

    nn.push_back(move(layer));
}  //----- end of addLayer


void FCNN::copyWeights(const FCNN &src)
// Algorithm : Check every layer for compatibility before copying parameters.
{
    if (nnInputSize != src.nnInputSize || nn.size() != src.nn.size())
    {
        throw invalid_argument("Incompatible dense networks");
    }

    for (size_t index = 0; index < nn.size(); index++)
    {
        if (nn[index].type != src.nn[index].type ||
            nn[index].outputSize != src.nn[index].outputSize)
        {
            throw invalid_argument("Incompatible dense layers");
        }
    }

    for (size_t index = 0; index < nn.size(); index++)
    {
        nn[index].weights = src.nn[index].weights;
        nn[index].biases = src.nn[index].biases;
    }
}  //----- end of copyWeights


vector<float> FCNN::forwardPropagation(const vector<float> &input)
// Algorithm : Evaluate each dense layer in order and replace the cached sample.
{
    if (nn.empty())
    {
        throw logic_error("FCNN has no layers");
    }

    if (input.size() != static_cast<size_t>(nnInputSize))
    {
        throw invalid_argument("Wrong FCNN input size");
    }

    const vector<float> *activation = &input;
    for (auto &layer : nn)
    {
        computeLayer(layer, *activation);
        activation = &layer.output;
    }

    return nn.back().output;
}  //----- end of forwardPropagation


vector<float> FCNN::backPropagation(const vector<float> &error)
// Algorithm : Apply each activation derivative, then multiply by the weights to propagate to the preceding input.
{
    if (nn.empty() || nn.back().output.empty())
    {
        throw logic_error("FCNN backward pass requires a forward pass");
    }

    if (error.size() != nn.back().output.size())
    {
        throw invalid_argument("Wrong FCNN output gradient size");
    }

    vector<float> gradient = error;
    for (auto layer = nn.rbegin(); layer != nn.rend(); ++layer)
    {
        layer->error = move(gradient);
        if (layer->type == "ReLU")
        {
            for (int out = 0; out < layer->outputSize; out++)
            {
                if (layer->output[out] <= 0.f)
                {
                    layer->error[out] = 0.f;
                }
            }
        }

        gradient.assign(layer->inputSize, 0.f);
        for (int in = 0; in < layer->inputSize; in++)
        {
            for (int out = 0; out < layer->outputSize; out++)
            {
                gradient[in] += layer->weights[in][out] * layer->error[out];
            }
        }
    }

    return gradient;
}  //----- end of backPropagation


void FCNN::resetGradients()
// Algorithm : Clear the batch gradients for all dense layers.
{
    for (auto &layer : nn)
    {
        for (auto &row : layer.weightGradient)
        {
            fill(row.begin(), row.end(), 0.f);
        }

        fill(layer.biasGradient.begin(), layer.biasGradient.end(), 0.f);
    }
}  //----- end of resetGradients


void FCNN::accumulateGradients()
// Algorithm : Add the cached input and preactivation gradient products to the batch.
{
    if (nn.empty() || nn.back().error.empty())
    {
        throw logic_error("FCNN gradient accumulation requires a backward pass");
    }

    for (auto &layer : nn)
    {
        for (int out = 0; out < layer.outputSize; out++)
        {
            for (int in = 0; in < layer.inputSize; in++)
            {
                layer.weightGradient[in][out] += layer.input[in] * layer.error[out];
            }

            layer.biasGradient[out] += layer.error[out];
        }
    }
}  //----- end of accumulateGradients


void FCNN::updateWeights(float learningRate)
// Algorithm : Apply the accumulated gradient to every dense weight and bias.
{
    for (auto &layer : nn)
    {
        for (int out = 0; out < layer.outputSize; out++)
        {
            for (int in = 0; in < layer.inputSize; in++)
            {
                layer.weights[in][out] -= learningRate * layer.weightGradient[in][out];
            }

            layer.biases[out] -= learningRate * layer.biasGradient[out];
        }
    }
}  //----- end of updateWeights


//---------------------------------------------------------------- PRIVATE
void FCNN::computeLayer(FullyConnectedLayer &layer, const vector<float> &input)
// Algorithm : Cache the input and apply the affine transform followed by the configured activation.
{
    layer.input = input;
    layer.error.clear();
    layer.output = layer.biases;

    for (int in = 0; in < layer.inputSize; in++)
    {
        for (int out = 0; out < layer.outputSize; out++)
        {
            layer.output[out] += input[in] * layer.weights[in][out];
        }
    }

    if (layer.type == "ReLU")
    {
        for (float &value : layer.output)
        {
            value = max(0.f, value);
        }
    }
}  //----- end of computeLayer
