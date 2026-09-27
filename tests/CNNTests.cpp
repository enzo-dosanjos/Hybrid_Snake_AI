/*************************************************************************
CNNTests - Convolution, pooling and numerical gradients
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//----- Implementation of the module <CNNTests> (file CNNTests.cpp) ------

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <algorithm>
#include <random>

using namespace std;

//------------------------------------------------------- Personal Include
#include "Test.h"
#include "src/DQN/CNN.h"


//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
Matrix<vector<float>> makeInput(int channels, int height, int width)
// Algorithm : Builds deterministic channel and spatial variations so tensor indexing errors affect the expected values.
{
    Matrix<vector<float>> input;
    input.resize(channels, vector<vector<float>>(height, vector<float>(width)));

    for (int channel = 0; channel < channels; channel++)
    {
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                input[channel][y][x] = 0.2f * (1 + channel + 2*y - x);
            }
        }
    }

    return input;
}  //----- end of makeInput


Matrix<vector<float>> referenceForward(const vector<ConvolutionalLayer> &layers,
                                      Matrix<vector<float>> input)
// Algorithm : Computes independent convolution outputs directly from layer values without calling production convolution.
{
    for (const auto &layer : layers)
    {
        int height = (input[0].size() + 2*layer.padding - layer.kernelSize) / layer.stride + 1;
        int width = (input[0][0].size() + 2*layer.padding - layer.kernelSize) / layer.stride + 1;
        Matrix<vector<float>> output;
        output.resize(layer.outputChannels, vector<vector<float>>(height, vector<float>(width)));

        for (int out = 0; out < layer.outputChannels; out++)
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    float value = layer.biases[out];

                    for (int in = 0; in < layer.inputChannels; in++)
                    {
                        for (int ky = 0; ky < layer.kernelSize; ky++)
                        {
                            for (int kx = 0; kx < layer.kernelSize; kx++)
                            {
                                // Map the output cell and kernel offset back into the unpadded input.
                                int iy = y*layer.stride + ky - layer.padding;
                                int ix = x*layer.stride + kx - layer.padding;

                                // Out-of-bounds samples contribute zero, avoiding the production padding helper.
                                if (iy >= 0 && iy < static_cast<int>(input[in].size()) &&
                                    ix >= 0 && ix < static_cast<int>(input[in][0].size()))
                                {
                                    value += input[in][iy][ix] * layer.weights[in][out][ky][kx];
                                }
                            }
                        }
                    }

                    output[out][y][x] = layer.type == "ReLU" ? max(0.f, value) : value;
                }
            }
        }

        input = output;
    }

    return input;
}  //----- end of referenceForward


float referenceLoss(const vector<ConvolutionalLayer> &layers, const Matrix<vector<float>> &input)
// Algorithm : Computes the summed half-squared error of reference outputs against a constant target of 0.7.
{
    auto output = referenceForward(layers, input);
    float loss = 0.f;

    for (const auto &channel : output)
    {
        for (const auto &row : channel)
        {
            for (float value : row)
            {
                loss += 0.5f * (value - 0.7f) * (value - 0.7f);
            }
        }
    }

    return loss;
}  //----- end of referenceLoss


void checkTensor(const Matrix<vector<float>> &actual, const Matrix<vector<float>> &expected)
// Algorithm : Checks tensor dimensions and each output value against the independent reference tensor.
{
    check(actual.size() == expected.size(), "Channel count differs");

    for (size_t channel = 0; channel < expected.size(); channel++)
    {
        check(actual[channel].size() == expected[channel].size(), "Tensor height differs");

        for (size_t y = 0; y < expected[channel].size(); y++)
        {
            check(actual[channel][y].size() == expected[channel][y].size(), "Tensor width differs");

            for (size_t x = 0; x < expected[channel][y].size(); x++)
            {
                checkNear(actual[channel][y][x], expected[channel][y][x], "Convolution value");
            }
        }
    }
}  //----- end of checkTensor


void testConvolutionShapes()
// Algorithm : Tests multi-channel convolution against reference values on rectangular inputs with varied padding and stride.
{
    for (int padding : {0, 1})
    {
        for (int stride : {1, 2})
        {
            CNN network(2, mt19937(42));
            int height = (5 + 2*padding - 3) / stride + 1;
            int width = (7 + 2*padding - 3) / stride + 1;
            network.addLayer("ReLU", 3, stride, padding, 3, {height, width}, 2);
            auto input = makeInput(2, 5, 7);
            auto expected = referenceForward(network.getNN(), input);

            checkTensor(network.forwardPropagation(input), expected);
        }
    }
}  //----- end of testConvolutionShapes


void testPooling()
// Algorithm : Tests channel-wise average pooling against analytical means across several channel counts and spatial shapes.
{
    for (int channels : {1, 2, 4})
    {
        for (auto shape : vector<pair<int, int>>{{1, 1}, {2, 5}, {5, 2}})
        {
            auto input = makeInput(channels, shape.first, shape.second);
            auto output = CNN::globalAvgPooling(input);
            check(output.size() == static_cast<size_t>(channels), "Pooling changed channel count");

            for (int channel = 0; channel < channels; channel++)
            {
                float expected = 0.2f * (1 + channel + (shape.first - 1) - (shape.second - 1)/2.f);
                checkNear(output[channel], expected, "Average pooling value");
            }
        }
    }
}  //----- end of testPooling


void testNumericalGradients()
// Algorithm : Tests every weight and bias gradient against finite differences, then checks that gradient reset clears them.
{
    // This seed and the input below keep the final ReLU active, away from its nondifferentiable boundary.
    CNN network(2, mt19937(4));
    network.addLayer("ReLU", 2, 1, 0, 2, {3, 4}, 2);
    network.addLayer("ReLU", 1, 1, 0, 1, {3, 4}, 2);
    auto input = makeInput(2, 4, 5);

    for (int channel = 0; channel < 2; channel++)
    {
        for (int y = 0; y < 4; y++)
        {
            for (int x = 0; x < 5; x++)
            {
                input[channel][y][x] = 0.5f + 0.03f*channel + 0.01f*y + 0.005f*x;
            }
        }
    }

    auto expectedOutput = referenceForward(network.getNN(), input);

    for (const auto &row : expectedOutput[0])
    {
        for (float value : row)
        {
            check(value > 0.001f, "Gradient fixture is too close to a ReLU boundary");
        }
    }

    auto error = network.forwardPropagation(input);

    for (auto &channel : error)
    {
        for (auto &row : channel)
        {
            for (float &value : row)
            {
                // Derivative of 0.5 * (output - target)^2 with respect to the output.
                value -= 0.7f;
            }
        }
    }

    network.resetGradients();
    network.backPropagation(error);
    network.accumulateGradient();
    auto layers = network.getNN();
    const float step = 1e-4f;
    float totalGradient = 0.f;

    for (size_t index = 0; index < layers.size(); index++)
    {
        const auto &layer = layers[index];

        for (int out = 0; out < layer.outputChannels; out++)
        {
            auto plus = layers;
            auto minus = layers;
            plus[index].biases[out] += step;
            minus[index].biases[out] -= step;
            // Central differences estimate one derivative while every other parameter remains fixed.
            float expected = (referenceLoss(plus, input) - referenceLoss(minus, input)) / (2*step);
            checkNear(layer.biasGradient[out], expected, "CNN bias gradient", 0.004f, 0.02f);

            for (int in = 0; in < layer.inputChannels; in++)
            {
                for (int y = 0; y < layer.kernelSize; y++)
                {
                    for (int x = 0; x < layer.kernelSize; x++)
                    {
                        // Reset both copies so each estimate perturbs only this weight.
                        plus = layers;
                        minus = layers;
                        plus[index].weights[in][out][y][x] += step;
                        minus[index].weights[in][out][y][x] -= step;
                        expected = (referenceLoss(plus, input) - referenceLoss(minus, input)) / (2*step);
                        float actual = layer.weightGradient[in][out][y][x];
                        checkNear(actual, expected, "CNN weight gradient", 0.004f, 0.02f);
                        totalGradient += abs(actual);
                    }
                }
            }
        }
    }

    check(totalGradient > 1e-5f, "Gradient fixture did not activate any weights");
    network.resetGradients();

    for (const auto &layer : network.getNN())
    {
        for (const auto &inputChannel : layer.weightGradient)
        {
            for (const auto &outputChannel : inputChannel)
            {
                for (const auto &row : outputChannel)
                {
                    for (float value : row)
                    {
                        checkNear(value, 0.f, "CNN gradient reset");
                    }
                }
            }
        }

        for (float value : layer.biasGradient)
        {
            checkNear(value, 0.f, "CNN bias reset");
        }
    }
}  //----- end of testNumericalGradients


void testWeightCopy()
// Algorithm : Tests weight copying by comparing outputs of differently seeded networks after copying their parameters.
{
    CNN first(2, mt19937(42));
    CNN second(2, mt19937(99));
    first.addLayer("ReLU", 1, 1, 0, 2, {3, 5}, 2);
    second.addLayer("ReLU", 1, 1, 0, 2, {3, 5}, 2);
    auto input = makeInput(2, 3, 5);
    second.copyWeights(first);
    checkTensor(first.forwardPropagation(input), second.forwardPropagation(input));
}  //----- end of testWeightCopy


vector<TestCase> getTests()
// Algorithm : Registers this component’s named test functions for independent selection and execution.
{
    return {
        {"convolution_shapes", testConvolutionShapes},
        {"pooling", testPooling},
        {"numerical_gradients", testNumericalGradients},
        {"weight_copy", testWeightCopy}
    };
}  //----- end of getTests
