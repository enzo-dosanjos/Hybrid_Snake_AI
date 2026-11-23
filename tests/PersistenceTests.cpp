/*************************************************************************
PersistenceTests - Model restoration and resumed learning
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//- Implementation of the module <PersistenceTests> (file PersistenceTests.cpp) -

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <fstream>
#include <iterator>
#include <limits>

using namespace std;

//------------------------------------------------------- Personal Include
#include "DQNTest.h"
#include "src/DQN/ModelPersistence.h"


//---------------------------------------------------------------- PRIVATE
static string readCheckpoint(const string &filename)
// Algorithm : Capture the serialized learner so tests can compare all persisted state.
{
    ifstream file(filename, ios::binary);
    check(file.is_open(), "Could not read test checkpoint");
    return {istreambuf_iterator<char>(file), istreambuf_iterator<char>()};
}  //----- end of readCheckpoint


static void checkLoadRejected(DQN &learner, const string &filename)
// Algorithm : Compare checkpoint snapshots around a failed load to verify no persisted state changed.
{
    ModelPersistence persistence;
    persistence.saveDQN(learner, "before.bin", 9);
    bool rejected = false;
    try
    {
        persistence.loadDQN(learner, filename);
    }
    catch (const exception &)
    {
        rejected = true;
    }

    check(rejected, "Invalid checkpoint was accepted: " + filename);
    persistence.saveDQN(learner, "after.bin", 9);
    check(readCheckpoint("before.bin") == readCheckpoint("after.bin"),
          "A rejected checkpoint changed learner state");
}  //----- end of checkLoadRejected


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
    config.replayCapacity = 16;
    DQN original(config, mt19937(42));
    populateReplay(original, state);
    // Exercise both supported representations of an absent next observation.
    original.remember({state, Action::Down, 0.25f, {}, true});
    original.remember({state, Action::Down, 0.5f, {}, false});
    original.train(5, 0.001f);
    // Save distinct online and target weights so copying only the online network cannot pass continuation checks.
    original.updateTargetNetworks();
    original.train(5, 0.001f);

    ModelPersistence persistence;
    persistence.saveDQN(original, "model.bin", 2);
    config.epsilon = 0.f;
    DQN restored(config, mt19937(99));
    populateReplay(restored, state);
    check(persistence.loadDQN(restored, "model.bin") == 2, "Wrong resumed training step");
    persistence.saveDQN(restored, "restored.bin", 2);
    check(readCheckpoint("model.bin") == readCheckpoint("restored.bin"),
          "Round trip changed the serialized learner or the replay order");

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
// Algorithm : Reject missing, malformed and late-truncated checkpoints without changing a populated learner.
{
    GameEngine engine({5, 4, 2, 2, 1}, {{0, 0}, {1, 1}, {4, 3}});
    auto state = encodeState(engine, 1);
    auto config = makeModelConfig(state.extraFeatures.size());
    DQN learner(config, mt19937(42));
    DQN other(config, mt19937(13));
    populateReplay(learner, state);
    other.remember({state, Action::Down, 1.f, {}, true});

    ModelPersistence persistence;
    persistence.saveDQN(other, "valid.bin", 7);
    const string contents = readCheckpoint("valid.bin");
    ofstream("invalid.bin", ios::binary) << "not a model";
    ofstream("truncated.bin", ios::binary) << contents.substr(0, contents.size() - 3);
    ofstream("trailing.bin", ios::binary) << contents << "unexpected";

    for (const char *filename : {"missing.bin", "invalid.bin", "truncated.bin", "trailing.bin"})
    {
        checkLoadRejected(learner, filename);
    }
}  //----- end of testInvalidFiles


void testIncompatibleConfiguration()
// Algorithm : Reject different feature groups with the same count, different network geometry and old schema metadata.
{
    ModelConfig config;
    config.features.snakeSize = false;
    config.extraFeatureCount = StateAnalyzer::getFeatureCount(config.numPlayers, config.features);
    DQN original(config, mt19937(1));
    ModelPersistence persistence;
    persistence.saveDQN(original, "model.bin", 0);

    auto otherFeatures = config;
    otherFeatures.features.snakeSize = true;
    otherFeatures.features.gameProgress = false;
    check(StateAnalyzer::getFeatureCount(config.numPlayers, otherFeatures.features) == config.extraFeatureCount,
          "Feature compatibility fixture must preserve the input size");
    DQN differentFeatures(otherFeatures, mt19937(2));
    checkLoadRejected(differentFeatures, "model.bin");

    auto otherGeometry = config;
    otherGeometry.convolutionLayers.front().stride++;
    DQN differentGeometry(otherGeometry, mt19937(2));
    checkLoadRejected(differentGeometry, "model.bin");

    const string contents = readCheckpoint("model.bin");
    const auto schemaStart = contents.find('\n') + 1;
    string olderObservation = contents;
    olderObservation.replace(schemaStart, olderObservation.find(' ', schemaStart) - schemaStart, "1");
    ofstream("old-observation.bin", ios::binary) << olderObservation;
    checkLoadRejected(original, "old-observation.bin");

    const auto featureStart = contents.find(' ', schemaStart) + 1;
    string olderFeatures = contents;
    olderFeatures.replace(featureStart, olderFeatures.find(' ', featureStart) - featureStart, "1");
    ofstream("old-features.bin", ios::binary) << olderFeatures;
    checkLoadRejected(original, "old-features.bin");
}  //----- end of testIncompatibleConfiguration


void testFailedSavePreservesCheckpoint()
// Algorithm : Force serialization to fail late and verify the previous checkpoint remains intact.
{
    GameEngine engine({5, 4, 2, 2, 1}, {{0, 0}, {1, 1}, {4, 3}});
    auto state = encodeState(engine, 1);
    DQN learner(makeModelConfig(state.extraFeatures.size()), mt19937(42));
    ModelPersistence persistence;
    persistence.saveDQN(learner, "model.bin", 5);
    const string saved = readCheckpoint("model.bin");
    learner.remember({state, Action::Down, numeric_limits<float>::infinity(), {}, true});

    bool rejected = false;
    try
    {
        persistence.saveDQN(learner, "model.bin", 6);
    }
    catch (const exception &)
    {
        rejected = true;
    }

    check(rejected && readCheckpoint("model.bin") == saved,
          "Failed save replaced a valid checkpoint");
}  //----- end of testFailedSavePreservesCheckpoint


vector<TestCase> getTests()
// Algorithm : Registers this component’s named test functions for independent selection and execution.
{
    return {
        {"round_trip", testRoundTrip},
        {"resumed_training", testResumedTraining},
        {"invalid_files", testInvalidFiles},
        {"incompatible_configuration", testIncompatibleConfiguration},
        {"failed_save_preserves_checkpoint", testFailedSavePreservesCheckpoint}
    };
}  //----- end of getTests
