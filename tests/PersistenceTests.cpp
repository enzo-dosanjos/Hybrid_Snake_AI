/*************************************************************************
PersistenceTests - Model restoration and resumed learning
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//- Implementation of the module <PersistenceTests> (file PersistenceTests.cpp) -

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <fstream>

using namespace std;

//------------------------------------------------------- Personal Include
#include "DQNTest.h"
#include "src/DQN/ModelPersistence.h"


//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
void testRoundTrip()
// Algorithm : Tests saved model restoration by checking the training step and predictions on two different states.
{
    GameEngine engine({5, 4, 2, 2, 1}, {{0, 0}, {1, 1}, {4, 3}});
    auto firstState = encodeState(engine, 1);
    engine.updateStep(Action::Down);
    auto secondState = encodeState(engine, 1);
    auto config = makeModelConfig(firstState.extraFeatures.size());
    DQN original(config, mt19937(42));
    DQN restored(config, mt19937(99));
    populateReplay(original, firstState);
    original.train(5, 0.001f);
    ModelPersistence persistence;
    persistence.saveDQN(original, "model.bin", 17);

    check(persistence.loadDQN(restored, "model.bin") == 17, "Training step was not restored");
    checkPredictions(original, restored, firstState);
    checkPredictions(original, restored, secondState);
}  //----- end of testRoundTrip


void testResumedTraining()
// Algorithm : Tests full training restoration by comparing exploration sequences and subsequent replay updates with an uninterrupted model.
{
    GameEngine engine({5, 4, 2, 2, 1}, {{0, 0}, {1, 1}, {4, 3}});
    auto state = encodeState(engine, 1);
    auto config = makeModelConfig(state.extraFeatures.size());
    DQN original(config, mt19937(42));
    populateReplay(original, state);
    original.train(5, 0.001f);
    // Save distinct online and target weights so copying only the online network cannot pass continuation checks.
    original.updateTargetNetworks();
    original.train(5, 0.001f);

    ModelPersistence persistence;
    persistence.saveDQN(original, "model.bin", 2);
    DQN restored(config, mt19937(99));
    check(persistence.loadDQN(restored, "model.bin") == 2, "Wrong resumed training step");

    for (int attempt = 0; attempt < 64; attempt++)
    {
        check(original.selectAction(state, true) == restored.selectAction(state, true),
              "Exploration generator state was not restored");
    }

    for (int update = 0; update < 8; update++)
    {
        original.train(5, 0.001f);
        restored.train(5, 0.001f);
        checkPredictions(original, restored, state);

        if (update % 3 == 0)
        {
            original.updateTargetNetworks();
            restored.updateTargetNetworks();
        }
    }
}  //----- end of testResumedTraining


void testInvalidFiles()
// Algorithm : Tests safe load failure by rejecting missing and malformed files without changing the model’s predictions.
{
    GameEngine engine({5, 4, 2, 2, 1}, {{0, 0}, {1, 1}, {4, 3}});
    auto state = encodeState(engine, 1);
    auto config = makeModelConfig(state.extraFeatures.size());
    DQN learner(config, mt19937(42));
    DQN unchanged(config, mt19937(42));
    ModelPersistence persistence;
    ofstream invalid("invalid.bin", ios::binary);
    invalid << "not a model";
    invalid.close();

    for (const string &filename : {"missing.bin", "invalid.bin"})
    {
        bool rejected = false;

        try
        {
            persistence.loadDQN(learner, filename);
        }
        catch (const exception &)
        {
            rejected = true;
        }

        check(rejected, "Invalid model file was accepted");
        checkPredictions(learner, unchanged, state);
    }
}  //----- end of testInvalidFiles


vector<TestCase> getTests()
// Algorithm : Registers this component’s named test functions for independent selection and execution.
{
    return {
        {"round_trip", testRoundTrip},
        {"resumed_training", testResumedTraining},
        {"invalid_files", testInvalidFiles}
    };
}  //----- end of getTests
