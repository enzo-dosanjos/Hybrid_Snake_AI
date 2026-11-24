/*************************************************************************
ModelPersistence - Saves and restores DQN training checkpoints
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//-- Implementation of the module <ModelPersistence> (file ModelPersistence.cpp) --

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <unistd.h>

using namespace std;

//------------------------------------------------------- Personal Include
#include "ModelPersistence.h"

//---------------------------------------------------------------- PRIVATE
static string checkpointHeader(const ModelConfig &config)
// Algorithm : Encode the version and all compatibility requirements in a canonical order.
{
    ostringstream header;
    header.imbue(locale::classic());
    header << setprecision(numeric_limits<float>::max_digits10);
    header << "HybridSnakeDQN 1\n" << config.observationSchemaVersion << ' '
           << config.featureSchemaVersion << ' ' << config.numPlayers << ' '
           << config.extraFeatureCount << '\n';

    const auto &features = config.features;
    header << features.distances << ' ' << features.snakeSize << ' ' << features.gameProgress << ' '
           << features.mobility << ' ' << features.wallProximity << '\n'
           << config.replayCapacity << ' ' << config.gamma << ' ' << config.minEpsilon << ' '
           << config.epsilonDecay << '\n';

    header << config.convolutionLayers.size() << '\n';
    for (const auto &layer : config.convolutionLayers)
    {
        header << layer.type << ' ' << layer.kernelSize << ' ' << layer.stride << ' '
               << layer.padding << ' ' << layer.outputChannels << '\n';
    }

    header << config.denseHiddenSizes.size();
    for (int size : config.denseHiddenSizes)
    {
        header << ' ' << size;
    }

    return header.str() + '\n';
}  //----- end of checkpointHeader


template <typename T>
static T readValue(istream &file)
// Algorithm : Read one whitespace-delimited number and reject missing, invalid or nonfinite values.
{
    T value;
    if (!(file >> value))
    {
        throw runtime_error("Invalid or incomplete DQN checkpoint");
    }

    int next = file.peek();
    if (next != EOF && !isspace(next))
    {
        throw runtime_error("Invalid DQN checkpoint delimiter");
    }

    if constexpr (is_floating_point_v<T>)
    {
        if (!isfinite(value))
        {
            throw runtime_error("Nonfinite DQN checkpoint value");
        }
    }

    return value;
}  //----- end of readValue


static void writeValue(ostream &file, float value)
// Algorithm : Write one finite parameter, feature or reward at round-trip precision.
{
    if (!isfinite(value))
    {
        throw runtime_error("Cannot save a nonfinite DQN checkpoint value");
    }

    file << value << ' ';
}  //----- end of writeValue


template <typename T>
static void writeValues(ostream &file, const T &values)
// Algorithm : Traverse the configured parameter containers down to their scalar values.
{
    for (const auto &value : values)
    {
        if constexpr (is_same_v<decay_t<decltype(value)>, float>)
        {
            writeValue(file, value);
        }
        else
        {
            writeValues(file, value);
        }
    }
}  //----- end of writeValues


template <typename T>
static void readValues(istream &file, T &values)
// Algorithm : Fill the existing parameter shapes without taking their dimensions from the file.
{
    for (auto &value : values)
    {
        if constexpr (is_same_v<decay_t<decltype(value)>, float>)
        {
            value = readValue<float>(file);
        }
        else
        {
            readValues(file, value);
        }
    }
}  //----- end of readValues


static mt19937 readGenerator(istream &file)
// Algorithm : Validate the Mersenne Twister state words and cursor before restoring the generator.
{
    ostringstream state;
    state.imbue(locale::classic());
    bool nonzero = false;
    for (size_t index = 0; index < mt19937::state_size; index++)
    {
        auto word = readValue<long long>(file);
        if (word < 0 || word > static_cast<long long>(mt19937::max()))
        {
            throw runtime_error("Invalid DQN checkpoint generator state");
        }

        nonzero = nonzero || word != 0;
        state << word << ' ';
    }

    int cursor = readValue<int>(file);
    if (!nonzero || cursor < 0 || cursor > static_cast<int>(mt19937::state_size))
    {
        throw runtime_error("Invalid DQN checkpoint generator cursor or state");
    }

    state << cursor;
    istringstream input(state.str());
    input.imbue(locale::classic());
    mt19937 gen;
    if (!(input >> gen))
    {
        throw runtime_error("Cannot restore DQN checkpoint generator");
    }

    return gen;
}  //----- end of readGenerator


static ActionMask readMask(istream &file)
// Algorithm : Reject values that use bits outside the action mask.
{
    int mask = readValue<int>(file);
    if (mask < 0 || mask >= (1 << ACTIONS.size()))
    {
        throw runtime_error("Invalid DQN checkpoint action mask");
    }

    return ActionMask(mask);
}  //----- end of readMask


void ModelPersistence::writeState(ostream &file, const DQN &dqn,
                                  const EncodedState &state, bool allowEmpty)
// Algorithm : Write rectangular board dimensions, channel values and extras in model order.
{
    if (allowEmpty && state.board.empty() && state.extraFeatures.empty())
    {
        file << "0 0\n";
        return;
    }

    if (state.board.size() != static_cast<size_t>(dqn.cnn.nnInputChannels) ||
        state.extraFeatures.size() != static_cast<size_t>(dqn.config.extraFeatureCount) ||
        state.board.front().empty())
    {
        throw runtime_error("Cannot save replay with an incompatible observation");
    }

    size_t height = state.board.front().size();
    size_t width = state.board.front().front().size();
    if (width == 0 || height > numeric_limits<int>::max() || width > numeric_limits<int>::max())
    {
        throw runtime_error("Cannot save replay with invalid board dimensions");
    }

    dqn.cnn.validateSpatialSize(height, width);
    file << height << ' ' << width << ' ';
    for (const auto &channel : state.board)
    {
        if (channel.size() != height)
        {
            throw runtime_error("Cannot save replay with inconsistent channel heights");
        }

        for (const auto &row : channel)
        {
            if (row.size() != width)
            {
                throw runtime_error("Cannot save replay with inconsistent row widths");
            }

            writeValues(file, row);
        }
    }

    writeValues(file, state.extraFeatures);
    file << '\n';
}  //----- end of writeState


EncodedState ModelPersistence::readState(istream &file, const DQN &dqn, ActionMask mask, bool allowEmpty)
// Algorithm : Read a model-sized observation, growing rows as their values arrive to bound allocations on truncated input.
{
    EncodedState state;
    state.legalActions = mask;
    int height = readValue<int>(file);
    int width = readValue<int>(file);
    if (height == 0 && width == 0 && allowEmpty)
    {
        return state;
    }

    if (height <= 0 || width <= 0)
    {
        throw runtime_error("Invalid DQN checkpoint board dimensions");
    }

    dqn.cnn.validateSpatialSize(height, width);
    for (int channel = 0; channel < dqn.cnn.nnInputChannels; channel++)
    {
        Matrix<float> &plane = state.board.emplace_back();
        for (int y = 0; y < height; y++)
        {
            vector<float> &row = plane.emplace_back();
            for (int x = 0; x < width; x++)
            {
                row.push_back(readValue<float>(file));
            }
        }
    }

    state.extraFeatures.resize(dqn.config.extraFeatureCount);
    readValues(file, state.extraFeatures);
    return state;
}  //----- end of readState


//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
void ModelPersistence::saveDQN(const DQN &dqn, const string &filename, int trainingStep) const
// Algorithm : Write the complete checkpoint to a temporary sibling and rename it over the destination.
{
    if (trainingStep < 0)
    {
        throw invalid_argument("DQN checkpoint training step must be nonnegative");
    }

    string temporary = filename + ".tmp.XXXXXX";
    int descriptor = mkstemp(temporary.data());
    if (descriptor < 0)
    {
        throw runtime_error("Cannot create DQN checkpoint beside " + filename);
    }

    close(descriptor);
    try
    {
        ofstream file(temporary, ios::binary);
        file.exceptions(ios::badbit | ios::failbit);
        file.imbue(locale::classic());
        file << checkpointHeader(dqn.config)
             << setprecision(numeric_limits<float>::max_digits10) << trainingStep << ' ';
        writeValue(file, dqn.epsilon);
        file << '\n' << dqn.gen << '\n' << dqn.replayBuffer.gen << '\n';

#ifdef SNAKE_LIBTORCH
        if (!dqn.onlineTorch.empty())
        {
            for (const auto *network : {&dqn.onlineTorch, &dqn.targetTorch})
            {
                for (const auto &row : network->checkpointRows())
                {
                    writeValues(file, row);
                    file << '\n';
                }
            }
        }
        else
#endif
        {
            auto writeNetwork = [&file](const auto &layers)
            {
                for (const auto &layer : layers)
                {
                    writeValues(file, layer.weights);
                    writeValues(file, layer.biases);
                    file << '\n';
                }
            };
            writeNetwork(dqn.cnn.nn);
            writeNetwork(dqn.fcnn.nn);
            writeNetwork(dqn.targetCNN.nn);
            writeNetwork(dqn.targetFCNN.nn);
        }

        file << dqn.replayBuffer.size() << '\n';
        for (const auto &transition : dqn.replayBuffer.buffer)
        {
            file << static_cast<int>(transition.action) << ' ' << transition.done << ' '
                 << transition.state.legalActions.to_ulong() << ' '
                 << transition.nextState.legalActions.to_ulong() << ' ';
            writeValue(file, transition.reward);
            file << '\n';
            writeState(file, dqn, transition.state, false);
            writeState(file, dqn, transition.nextState,
                       transition.done || transition.nextState.legalActions.none());
        }

        file.close();
        filesystem::rename(temporary, filename);
    }
    catch (...)
    {
        error_code ignored;
        filesystem::remove(temporary, ignored);
        throw;
    }
}  //----- end of saveDQN


int ModelPersistence::loadDQN(DQN &dqn, const string &filename) const
// Algorithm : Verify compatibility and decode all contents into a separate learner before publishing it.
{
    ifstream file(filename, ios::binary);
    file.imbue(locale::classic());
    if (!file)
    {
        throw runtime_error("Cannot open DQN checkpoint " + filename);
    }

    for (char expected : checkpointHeader(dqn.config))
    {
        if (file.get() != expected)
        {
            throw runtime_error("Incompatible DQN checkpoint header");
        }
    }

    int trainingStep = readValue<int>(file);
    float epsilon = readValue<float>(file);
    if (trainingStep < 0 || epsilon < 0.f || epsilon > 1.f)
    {
        throw runtime_error("Invalid DQN checkpoint training step or exploration rate");
    }

    // Caches and gradients are rebuilt by the next prediction or training batch.
    DQN restored(dqn.config, mt19937{});
    restored.epsilon = epsilon;
    restored.gen = readGenerator(file);
    restored.replayBuffer.gen = readGenerator(file);

    auto readNetwork = [&file](auto &layers)
    {
        for (auto &layer : layers)
        {
            readValues(file, layer.weights);
            readValues(file, layer.biases);
        }
    };
    readNetwork(restored.cnn.nn);
    readNetwork(restored.fcnn.nn);
    readNetwork(restored.targetCNN.nn);
    readNetwork(restored.targetFCNN.nn);

    int count = readValue<int>(file);
    if (count < 0 || count > restored.replayBuffer.capacity)
    {
        throw runtime_error("Invalid DQN checkpoint replay size");
    }

    for (int index = 0; index < count; index++)
    {
        int action = readValue<int>(file);
        int done = readValue<int>(file);
        if (action < 0 || action >= static_cast<int>(ACTIONS.size()) || (done != 0 && done != 1))
        {
            throw runtime_error("Invalid DQN checkpoint replay action or terminal flag");
        }

        Transition &transition = restored.replayBuffer.buffer.emplace_back();
        transition.action = static_cast<Action>(action);
        transition.done = done;
        ActionMask stateMask = readMask(file);
        ActionMask nextMask = readMask(file);
        transition.reward = readValue<float>(file);
        transition.state = readState(file, restored, stateMask, false);
        transition.nextState = readState(file, restored, nextMask, transition.done || nextMask.none());
    }

    file >> ws;
    if (!file.eof() || file.bad())
    {
        throw runtime_error("Unexpected data after DQN checkpoint");
    }

    static_assert(is_nothrow_move_assignable_v<DQN>, "Checkpoint publication must not throw");
    dqn = move(restored);
    return trainingStep;
}  //----- end of loadDQN
