/*************************************************************************
NeuralNetwork - Shared gradient operations for trainable networks
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//------ Interface of the module <NeuralNetwork> (file NeuralNetwork.h) ------
#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

//------------------------------------------------------------------------
// Role of the <NeuralNetwork> module
// Defines the batch training operations common to networks with different
// input and output representations.
//------------------------------------------------------------------------

//----------------------------------------------------------------- PUBLIC
class NeuralNetwork {
//--------------------------------------------------------- Public Methods
    public:
        virtual void resetGradients() = 0;
        // Contract : Clear the accumulated weight and bias gradients for every layer.

        virtual void accumulateGradients() = 0;
        // Contract : Add the gradients of the latest forward and backward pass to the batch.

        virtual void updateWeights(float learningRate) = 0;
        // Contract : Subtract learningRate times the accumulated gradients from the parameters.
        // The caller can divide learningRate by the batch size to apply a mean gradient.

//---------------------------------------------- Constructors - destructor
        virtual ~NeuralNetwork() = default;
};

#endif //NEURALNETWORK_H
