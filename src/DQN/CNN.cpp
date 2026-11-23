/*************************************************************************
CNN - A convolution neural network
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//--------- Implementation of the module <CNN> (file CNN.cpp) ---------

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

using namespace std;

//------------------------------------------------------- Personal Include
#include "CNN.h"

//---------------------------------------------------------------- PRIVATE
static pair<int, int> spatialSize(const FeatureTensor &input)
// Algorithm : Validate a nonempty rectangular channels-first tensor and return its height and width.
{
    if (input.empty() || input.front().empty() || input.front().front().empty())
    {
        throw invalid_argument("CNN tensor must have positive dimensions");
    }

    size_t height = input.front().size();
    size_t width = input.front().front().size();
    if (height > numeric_limits<int>::max() || width > numeric_limits<int>::max())
    {
        throw invalid_argument("CNN tensor is too large");
    }

    for (const auto &channel : input)
    {
        if (channel.size() != height)
        {
            throw invalid_argument("CNN tensor channels must have the same height");
        }

        for (const auto &row : channel)
        {
            if (row.size() != width)
            {
                throw invalid_argument("CNN tensor rows must have the same width");
            }
        }
    }

    return {height, width};
}  //----- end of spatialSize


//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
CNN::CNN(int inputChannels, const mt19937 &p_gen) : nnInputChannels(inputChannels), gen(p_gen)
// Algorithm : Store the channel count and random generator used to initialize the convolution kernels.
{
    if (inputChannels <= 0)
    {
        throw invalid_argument("CNN input channel count must be positive");
    }
}  //----- end of CNN


void CNN::addLayer(const ConvLayerConfig &config)
// Algorithm : Initialize a convolutional layer whose input channels match the preceding output.
{
    if ((config.type != "ReLU" && config.type != "linear") ||
        config.kernelSize <= 0 || config.stride <= 0 ||
        config.padding < 0 || config.outputChannels <= 0)
    {
        throw invalid_argument("Invalid convolution layer configuration");
    }

    ConvolutionalLayer layer;
    layer.type = config.type;
    layer.inputChannels = nn.empty() ? nnInputChannels : nn.back().outputChannels;
    layer.outputChannels = config.outputChannels;
    layer.kernelSize = config.kernelSize;
    layer.stride = config.stride;
    layer.padding = config.padding;
    Matrix<float> kernel(layer.kernelSize, layer.kernelSize);
    layer.weights = Matrix<Matrix<float>>(layer.outputChannels, layer.inputChannels, kernel);
    layer.biases.resize(layer.outputChannels);
    layer.weightGradient = layer.weights;
    layer.biasGradient.resize(layer.outputChannels);

    uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (auto &inputChannel : layer.weights)
    {
        for (auto &outputChannel : inputChannel)
        {
            for (auto &row : outputChannel)
            {
                for (float &weight : row)
                {
                    weight = dist(gen);
                }
            }
        }
    }

    nn.push_back(move(layer));
}  //----- end of addLayer


void CNN::copyWeights(const CNN &src)
// Algorithm : Verify convolution geometry and activations before copying only weights and biases.
{
    if (nnInputChannels != src.nnInputChannels || nn.size() != src.nn.size())
    {
        throw invalid_argument("Incompatible convolutional networks");
    }

    for (size_t index = 0; index < nn.size(); index++)
    {
        const auto &destination = nn[index];
        const auto &source = src.nn[index];
        if (destination.type != source.type || destination.outputChannels != source.outputChannels ||
            destination.kernelSize != source.kernelSize ||
            destination.stride != source.stride || destination.padding != source.padding)
        {
            throw invalid_argument("Incompatible convolutional layers");
        }
    }

    for (size_t index = 0; index < nn.size(); index++)
    {
        nn[index].weights = src.nn[index].weights;
        nn[index].biases = src.nn[index].biases;
    }
}  //----- end of copyWeights


FeatureTensor CNN::forwardPropagation(const BoardTensor &input)
// Algorithm : Validate the spatial dimensions, then convolve each layer in order and cache the activations.
{
    if (nn.empty())
    {
        throw logic_error("CNN has no layers");
    }

    if (input.size() != static_cast<size_t>(nnInputChannels))
    {
        throw invalid_argument("Wrong CNN input channel count");
    }

    auto [height, width] = spatialSize(input);
    for (const auto &layer : nn)
    {
        long long paddedHeight = height + 2LL*layer.padding;
        long long paddedWidth = width + 2LL*layer.padding;
        if (paddedHeight < layer.kernelSize || paddedWidth < layer.kernelSize ||
            paddedHeight > numeric_limits<int>::max() || paddedWidth > numeric_limits<int>::max())
        {
            throw invalid_argument("Invalid padded dimensions for convolution");
        }

        height = (paddedHeight - layer.kernelSize) / layer.stride + 1;
        width = (paddedWidth - layer.kernelSize) / layer.stride + 1;
    }

    const FeatureTensor *activation = &input;
    for (auto &layer : nn)
    {
        convolution(layer, *activation);
        activation = &layer.output;
    }

    return nn.back().output;
}  //----- end of forwardPropagation


vector<float> CNN::globalAvgPooling(const FeatureTensor &input)
// Algorithm : Divide each channel's sum by the spatial area of the feature maps.
{
    if (input.empty())
    {
        return {};
    }

    auto [height, width] = spatialSize(input);
    float area = static_cast<float>(height) * width;
    vector<float> output;
    output.reserve(input.size());

    for (const auto &channel : input)
    {
        float sum = 0.f;
        for (const auto &row : channel)
        {
            for (float value : row)
            {
                sum += value;
            }
        }

        output.push_back(sum / area);
    }

    return output;
}  //----- end of globalAvgPooling


FeatureTensor CNN::globalAvgPoolingBackward(const vector<float> &pooledGradient, const FeatureTensor &input)
// Algorithm : Divide each pooled gradient equally among the cells that contributed to that channel's mean.
{
    if (pooledGradient.size() != input.size())
    {
        throw invalid_argument("Wrong pooled gradient size");
    }

    if (input.empty())
    {
        return {};
    }

    auto [height, width] = spatialSize(input);
    float area = static_cast<float>(height) * width;
    FeatureTensor gradient;
    gradient.reserve(input.size());

    for (float value : pooledGradient)
    {
        gradient.emplace_back(width, height, value / area);
    }

    return gradient;
}  //----- end of globalAvgPoolingBackward


void CNN::backPropagation(const FeatureTensor &error)
// Algorithm : Gate each output gradient with its activation derivative and scatter it to the preceding layer.
{
    if (nn.empty() || nn.back().output.empty())
    {
        throw logic_error("CNN backward pass requires a forward pass");
    }

    auto [height, width] = spatialSize(error);
    const auto &output = nn.back().output;
    if (error.size() != output.size() ||
        static_cast<size_t>(height) != output.front().size() ||
        static_cast<size_t>(width) != output.front().front().size())
    {
        throw invalid_argument("Wrong CNN output gradient shape");
    }

    FeatureTensor gradient = error;
    for (size_t index = nn.size(); index > 0; index--)
    {
        auto &layer = nn[index - 1];
        layer.error = move(gradient);
        if (layer.type == "ReLU")
        {
            for (int out = 0; out < layer.outputChannels; out++)
            {
                for (size_t y = 0; y < layer.output[out].size(); y++)
                {
                    for (size_t x = 0; x < layer.output[out][y].size(); x++)
                    {
                        if (layer.output[out][y][x] <= 0.f)
                        {
                            layer.error[out][y][x] = 0.f;
                        }
                    }
                }
            }
        }

        if (index == 1)
        {
            break;
        }

        const auto &previousOutput = nn[index - 2].output;
        int previousHeight = previousOutput.front().size();
        int previousWidth = previousOutput.front().front().size();
        gradient.assign(layer.inputChannels, Matrix<float>(previousWidth, previousHeight));

        for (int out = 0; out < layer.outputChannels; out++)
        {
            for (int y = 0; y < static_cast<int>(layer.error[out].size()); y++)
            {
                for (int x = 0; x < static_cast<int>(layer.error[out][y].size()); x++)
                {
                    for (int ky = 0; ky < layer.kernelSize; ky++)
                    {
                        for (int kx = 0; kx < layer.kernelSize; kx++)
                        {
                            // Undo the padding offset to scatter into the preceding unpadded activation.
                            int inputY = y*layer.stride + ky - layer.padding;
                            int inputX = x*layer.stride + kx - layer.padding;
                            if (inputY >= 0 && inputY < previousHeight && inputX >= 0 && inputX < previousWidth)
                            {
                                for (int in = 0; in < layer.inputChannels; in++)
                                {
                                    gradient[in][inputY][inputX] +=
                                        layer.weights[in][out][ky][kx] * layer.error[out][y][x];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}  //----- end of backPropagation


void CNN::resetGradients()
// Algorithm : Clear all accumulated kernel and bias gradients.
{
    for (auto &layer : nn)
    {
        for (auto &inputChannel : layer.weightGradient)
        {
            for (auto &outputChannel : inputChannel)
            {
                for (auto &row : outputChannel)
                {
                    fill(row.begin(), row.end(), 0.f);
                }
            }
        }

        fill(layer.biasGradient.begin(), layer.biasGradient.end(), 0.f);
    }
}  //----- end of resetGradients


void CNN::accumulateGradients()
// Algorithm : Sum the cached padded inputs times their output gradients over the convolution windows.
{
    if (nn.empty() || nn.back().error.empty())
    {
        throw logic_error("CNN gradient accumulation requires a backward pass");
    }

    for (auto &layer : nn)
    {
        for (int out = 0; out < layer.outputChannels; out++)
        {
            for (size_t y = 0; y < layer.error[out].size(); y++)
            {
                for (size_t x = 0; x < layer.error[out][y].size(); x++)
                {
                    float error = layer.error[out][y][x];
                    layer.biasGradient[out] += error;

                    for (int in = 0; in < layer.inputChannels; in++)
                    {
                        for (int ky = 0; ky < layer.kernelSize; ky++)
                        {
                            for (int kx = 0; kx < layer.kernelSize; kx++)
                            {
                                layer.weightGradient[in][out][ky][kx] +=
                                    layer.input[in][y*layer.stride + ky][x*layer.stride + kx] * error;
                            }
                        }
                    }
                }
            }
        }
    }
}  //----- end of accumulateGradients


void CNN::updateWeights(float learningRate)
// Algorithm : Apply the accumulated gradient to every convolution kernel and bias.
{
    for (auto &layer : nn)
    {
        for (int out = 0; out < layer.outputChannels; out++)
        {
            for (int in = 0; in < layer.inputChannels; in++)
            {
                for (int ky = 0; ky < layer.kernelSize; ky++)
                {
                    for (int kx = 0; kx < layer.kernelSize; kx++)
                    {
                        layer.weights[in][out][ky][kx] -= learningRate * layer.weightGradient[in][out][ky][kx];
                    }
                }
            }

            layer.biases[out] -= learningRate * layer.biasGradient[out];
        }
    }
}  //----- end of updateWeights


//---------------------------------------------------------------- PRIVATE
FeatureTensor CNN::padInput(const FeatureTensor &input, int padding)
// Algorithm : Copy each channel into a zero-filled matrix with a border of the requested width.
{
    int height = input.front().size();
    int width = input.front().front().size();
    FeatureTensor padded(input.size(), Matrix<float>(width + 2*padding, height + 2*padding));

    for (size_t channel = 0; channel < input.size(); channel++)
    {
        for (int y = 0; y < height; y++)
        {
            copy(input[channel][y].begin(), input[channel][y].end(), padded[channel][y + padding].begin() + padding);
        }
    }

    return padded;
}  //----- end of padInput


void CNN::convolution(ConvolutionalLayer &layer, const FeatureTensor &input)
// Algorithm : Cache the padded input and evaluate each kernel window plus bias and activation.
{
    layer.input = padInput(input, layer.padding);
    int height = layer.input.front().size();
    int width = layer.input.front().front().size();
    int outputHeight = (height - layer.kernelSize) / layer.stride + 1;
    int outputWidth = (width - layer.kernelSize) / layer.stride + 1;
    layer.output.assign(layer.outputChannels, Matrix<float>(outputWidth, outputHeight));
    layer.error.clear();

    for (int out = 0; out < layer.outputChannels; out++)
    {
        for (int y = 0; y < outputHeight; y++)
        {
            for (int x = 0; x < outputWidth; x++)
            {
                float value = layer.biases[out];
                for (int in = 0; in < layer.inputChannels; in++)
                {
                    for (int ky = 0; ky < layer.kernelSize; ky++)
                    {
                        for (int kx = 0; kx < layer.kernelSize; kx++)
                        {
                            value += layer.input[in][y*layer.stride + ky][x*layer.stride + kx] *
                                     layer.weights[in][out][ky][kx];
                        }
                    }
                }

                layer.output[out][y][x] = layer.type == "ReLU" ? max(0.f, value) : value;
            }
        }
    }
}  //----- end of convolution
