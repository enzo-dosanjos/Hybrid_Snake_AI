/*************************************************************************
FCNNTests - Dense layers, gradients and batch accumulation
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//---- Implementation of the module <FCNNTests> (file FCNNTests.cpp) -----

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <algorithm>
#include <random>

using namespace std;

//------------------------------------------------------- Personal Include
#include "Test.h"
#include "src/DQN/FCNN.h"


//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
vector<float> referenceForward(const vector<FullyConnectedLayer> &layers, vector<float> input)
// Algorithm : Computes independent dense-layer outputs using explicit matrix sums and activation functions.
{
    for (const auto &layer : layers)
    {
        vector<float> output = layer.biases;

        for (int out = 0; out < layer.outputSize; out++)
        {
            for (int in = 0; in < layer.inputSize; in++)
            {
                output[out] += input[in] * layer.weights[in][out];
            }

            if (layer.type == "ReLU")
            {
                output[out] = max(0.f, output[out]);
            }
        }

        input = output;
    }

    return input;
}  //----- end of referenceForward


float referenceLoss(const vector<FullyConnectedLayer> &layers, const vector<float> &input)
// Algorithm : Computes the summed half-squared error of reference outputs against a constant target of one.
{
    float loss = 0.f;

    for (float value : referenceForward(layers, input))
    {
        loss += 0.5f * (value - 1.f) * (value - 1.f);
    }

    return loss;
}  //----- end of referenceLoss


void accumulate(FCNN &network, const vector<float> &input)
// Algorithm : Accumulates one sample’s gradients by backpropagating its output error against a target of one.
{
    vector<float> error = network.forwardPropagation(input);

    for (float &value : error)
    {
        value -= 1.f;
    }

    network.backPropagation(error);
    network.accumulateGradients();
}  //----- end of accumulate


void testForwardPropagation()
// Algorithm : Tests dense forward values against an independent calculation for zero, positive and negative inputs.
{
    FCNN network(3, mt19937(42));
    network.addLayer("ReLU", 5);
    network.addLayer("linear", 2);

    for (const auto &input : vector<vector<float>>{{0, 0, 0}, {1, -2, 3}, {-3, 2, -1}})
    {
        auto actual = network.forwardPropagation(input);
        auto expected = referenceForward(network.getNN(), input);
        check(actual.size() == expected.size(), "Dense output size differs");

        for (size_t index = 0; index < actual.size(); index++)
        {
            checkNear(actual[index], expected[index], "Dense forward value");
        }
    }
}  //----- end of testForwardPropagation


void testNumericalGradients()
// Algorithm : Checks the parameter and input gradients against independent finite differences.
{
    FCNN network(2, mt19937(42));
    network.addLayer("ReLU", 3);
    network.addLayer("linear", 2);
    vector<float> input{0.7f, -1.2f};
    auto error = network.forwardPropagation(input);
    for (float &value : error)
    {
        value -= 1.f;
    }

    auto inputGradient = network.backPropagation(error);
    network.accumulateGradients();
    auto layers = network.getNN();
    const float step = 1e-3f;
    float firstLayerGradient = 0.f;

    for (size_t index = 0; index < layers.size(); index++)
    {
        const auto &layer = layers[index];

        for (int out = 0; out < layer.outputSize; out++)
        {
            auto plus = layers;
            auto minus = layers;
            plus[index].biases[out] += step;
            minus[index].biases[out] -= step;
            // Central differences estimate the derivative independently of production backpropagation.
            float expected = (referenceLoss(plus, input) - referenceLoss(minus, input)) / (2*step);
            checkNear(layer.biasGradient[out], expected, "FCNN bias gradient", 0.001f, 0.01f);

            for (int in = 0; in < layer.inputSize; in++)
            {
                // Discard the previous perturbation before checking the next weight.
                plus = layers;
                minus = layers;
                plus[index].weights[in][out] += step;
                minus[index].weights[in][out] -= step;
                expected = (referenceLoss(plus, input) - referenceLoss(minus, input)) / (2*step);
                checkNear(layer.weightGradient[in][out], expected, "FCNN weight gradient", 0.001f, 0.01f);

                if (index == 0)
                {
                    firstLayerGradient += abs(layer.weightGradient[in][out]);
                }
            }
        }
    }

    check(firstLayerGradient > 1e-5f, "First layer never receives a gradient");
    check(inputGradient.size() == input.size(), "Input gradient size differs");
    for (size_t index = 0; index < input.size(); index++)
    {
        // Perturb one input feature while holding the weights fixed.
        auto plus = input;
        auto minus = input;
        plus[index] += step;
        minus[index] -= step;
        float expected = (referenceLoss(layers, plus) - referenceLoss(layers, minus)) / (2*step);
        checkNear(inputGradient[index], expected, "FCNN input gradient", 0.001f, 0.01f);
    }
}  //----- end of testNumericalGradients


void testOutputReluGradient()
// Algorithm : Tests that the final ReLU passes gradients only for positive preactivations.
{
    FCNN network(1, mt19937(42));
    network.addLayer("ReLU", 1);
    float weight = network.getNN()[0].weights[0][0];
    float sign = weight > 0.f ? 1.f : -1.f;

    network.forwardPropagation({-sign});
    checkNear(network.backPropagation({2.f})[0], 0.f, "Inactive output input gradient");
    network.accumulateGradients();

    network.forwardPropagation({sign});
    checkNear(network.backPropagation({2.f})[0], 2.f*weight, "Active output input gradient");
    network.accumulateGradients();

    const auto layer = network.getNN()[0];
    checkNear(layer.biasGradient[0], 2.f, "Inactive output contributed to the bias gradient");
    checkNear(layer.weightGradient[0][0], 2.f*sign, "Incorrect output weight gradient");
}  //----- end of testOutputReluGradient


void testAccumulationAndReset()
// Algorithm : Tests batch gradients against the sum of separate samples, then checks that the first layer resets.
{
    FCNN network(2, mt19937(42));
    network.addLayer("linear", 2);
    vector<float> first{1.f, -2.f};
    vector<float> second{-0.5f, 3.f};

    network.resetGradients();
    accumulate(network, first);
    auto firstGradient = network.getNN()[0];
    network.resetGradients();
    accumulate(network, second);
    auto secondGradient = network.getNN()[0];

    network.resetGradients();
    accumulate(network, first);
    accumulate(network, second);
    auto batch = network.getNN()[0];

    for (int out = 0; out < 2; out++)
    {
        checkNear(batch.biasGradient[out], firstGradient.biasGradient[out] + secondGradient.biasGradient[out],
                  "Batch bias gradient must sum both samples");

        for (int in = 0; in < 2; in++)
        {
            checkNear(batch.weightGradient[in][out],
                      firstGradient.weightGradient[in][out] + secondGradient.weightGradient[in][out],
                      "Batch weight gradient must sum both samples");
        }
    }

    network.resetGradients();
    auto cleared = network.getNN()[0];

    for (int out = 0; out < 2; out++)
    {
        checkNear(cleared.biasGradient[out], 0.f, "First-layer bias reset");

        for (int in = 0; in < 2; in++)
        {
            checkNear(cleared.weightGradient[in][out], 0.f, "First-layer weight reset");
        }
    }
}  //----- end of testAccumulationAndReset


void testUpdateAndCopy()
// Algorithm : Tests loss reduction and copy independence by updating one network and checking the unchanged copy.
{
    FCNN network(2, mt19937(42));
    FCNN copied(2, mt19937(99));
    network.addLayer("linear", 1);
    copied.addLayer("linear", 1);
    copied.copyWeights(network);
    vector<float> input{1.f, -2.f};
    auto before = network.getNN();
    checkNear(copied.forwardPropagation(input)[0], network.forwardPropagation(input)[0], "Copied output");

    network.resetGradients();
    accumulate(network, input);
    network.updateWeights(0.01f);

    check(referenceLoss(network.getNN(), input) < referenceLoss(before, input), "Gradient step increased loss");
    checkNear(copied.forwardPropagation(input)[0], referenceForward(before, input)[0],
              "Updating original changed its copy");
}  //----- end of testUpdateAndCopy


vector<TestCase> getTests()
// Algorithm : Registers this component’s named test functions for independent selection and execution.
{
    return {
        {"forward_propagation", testForwardPropagation},
        {"numerical_gradients", testNumericalGradients},
        {"output_relu_gradient", testOutputReluGradient},
        {"accumulation_and_reset", testAccumulationAndReset},
        {"update_and_copy", testUpdateAndCopy}
    };
}  //----- end of getTests
