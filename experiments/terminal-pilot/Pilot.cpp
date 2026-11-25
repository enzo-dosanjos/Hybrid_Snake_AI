/*************************************************************************
Pilot - Runs the terminal-reward training experiment
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//--------- Implementation of the module <Pilot> (file Pilot.cpp) ---------

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <algorithm>
#include <cmath>
#include <sstream>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace std;

//------------------------------------------------------- Personal Include
#include "../../src/Agent/MinimaxAgent.h"
#include "../../src/ApplicationModel.h"
#include "../../src/DQN/ModelPersistence.h"
#include "../../src/DQN/ResultValidation.h"
#include "../../src/GameEngine/InputHandler.h"
#include "../../src/Minimax/MinimaxParamsFile.h"
#include "../../src/TrainingLoop.h"
#include "RandomLegalAgent.h"

//------------------------------------------------------------------ Types
struct PilotConfig {
    int seed,
        shallowDepth,
        randomWeight,
        shallowWeight,
        tunedWeight;
    int decayUpdates,
        batchSize,
        targetEvery,
        saveEvery,
        evaluateEvery,
        threads;

    float minimumEpsilon, learningRate;
    string paramsFile;
};


//---------------------------------------------------------------- PRIVATE
static int parseInteger(const string &text, const string &name, int minimum)
// Algorithm : Read an integer argument in full and enforce its lower bound.
{
    size_t end = 0;
    int value = stoi(text, &end);
    if (end != text.size() || value < minimum)
    {
        throw invalid_argument("Invalid value for " + name);
    }

    return value;
}  //----- end of parseInteger


static ifstream openSetup(const string &filename)
// Algorithm : Open a caller-supplied setup file and report the path if it cannot be read.
{
    ifstream input(filename);
    if (!input)
    {
        throw runtime_error("Cannot open setup file " + filename);
    }

    return input;
}  //----- end of openSetup


static vector<ValidationMatch> readValidationSet(
    const string &filename, int numPlayers, const vector<reference_wrapper<Agent>> &opponents
)
// Algorithm : Validate each setup and attach the Minimax opponent to every other seat.
{
    ifstream input = openSetup(filename);
    vector<ValidationMatch> matches;

    while ((input >> ws).peek() != char_traits<char>::eof())
    {
        GameConfig rules = InputHandler::readGameConfig(input);
        if (rules.N != numPlayers)
        {
            throw invalid_argument("Training and validation setups must use the same number of players");
        }

        auto origins = InputHandler::readPlayersOrigins(rules.N, input);
        // Validate origins and board dimensions before storing the matchup.
        GameEngine(rules, origins);
        matches.push_back({rules, move(origins), opponents});
    }

    if (input.bad() || !input.eof() || matches.empty())
    {
        throw invalid_argument("Expected at least one complete validation setup in " + filename);
    }

    return matches;
}  //----- end of readValidationSet


template <typename T>
static void readSetting(istream &input, const string &name, T &value)
// Algorithm : Require each named setting in the order used by the checked-in configuration.
{
    string key;
    if (!(input >> key >> value) || key != name)
    {
        throw invalid_argument("Expected pilot setting " + name);
    }
}  //----- end of readSetting


static PilotConfig readConfig(const string &settings)
// Algorithm : Decode and validate all experiment settings before creating a learner.
{
    istringstream input(settings);
    PilotConfig config{};

    readSetting(input, "seed", config.seed);
    readSetting(input, "shallowDepth", config.shallowDepth);
    readSetting(input, "randomWeight", config.randomWeight);
    readSetting(input, "shallowWeight", config.shallowWeight);
    readSetting(input, "tunedWeight", config.tunedWeight);

    readSetting(input, "decayUpdates", config.decayUpdates);
    readSetting(input, "minimumEpsilon", config.minimumEpsilon);

    readSetting(input, "batchSize", config.batchSize);
    readSetting(input, "learningRate", config.learningRate);
    readSetting(input, "targetEvery", config.targetEvery);
    readSetting(input, "saveEvery", config.saveEvery);
    readSetting(input, "evaluateEvery", config.evaluateEvery);
    readSetting(input, "threads", config.threads);
    readSetting(input, "minimaxParams", config.paramsFile);

    if (!(input >> ws).eof() || config.randomWeight < 0 || config.shallowWeight < 0 ||
        config.tunedWeight < 0 || config.randomWeight + 0LL + config.shallowWeight + config.tunedWeight != 100 ||
        config.minimumEpsilon < 0.f || config.minimumEpsilon > 1.f || !isfinite(config.minimumEpsilon) ||
        !isfinite(config.learningRate) || config.learningRate <= 0.f ||
        min({config.decayUpdates, config.batchSize, config.targetEvery, config.saveEvery,
             config.evaluateEvery, config.threads, config.shallowDepth}) < 1)
    {
        throw invalid_argument("Invalid pilot settings; opponent weights must sum to 100");
    }

    return config;
}  //----- end of readConfig


static string readText(const string &path)
// Algorithm : Read a text fixture in full so resume can verify the original experiment inputs.
{
    auto input = openSetup(path);
    return string(istreambuf_iterator<char>(input), istreambuf_iterator<char>());
}  //----- end of readText


static void saveEpisode(const filesystem::path &output, int episode, int updates,
                        const string &settings, const DQN &learner)
// Algorithm : Publish the model and matching metadata together by renaming a completed directory.
{
    auto destination = output / ("episode-" + to_string(episode));
    auto pending = destination.string() + ".partial";
    if (filesystem::exists(destination) || !filesystem::create_directory(pending))
    {
        throw runtime_error("Pilot checkpoint already exists: " + destination.string());
    }

    try
    {
        ModelPersistence{}.saveDQN(learner, pending + "/model.model", updates);

        ofstream state;
        state.exceptions(ios::badbit | ios::failbit);
        state.open(pending + "/pilot.state");
        state << episode << ' ' << updates << '\n' << settings;
        state.close();

        // Readers can open the episode only after both files have been closed.
        filesystem::rename(pending, destination);
    }
    catch (...)
    {
        filesystem::remove_all(pending);
        throw;
    }
}  //----- end of saveEpisode


static void playEpisode(GameEngine &engine, DQN &learner, DQNAgent &greedy, RandomLegalAgent &random,
                        const vector<reference_wrapper<Agent>> &opponents, const PilotConfig &config,
                        mt19937 &generator, int episode, const string &opponent, int &updates)
// Algorithm : Sample the pilot policy at each decision and store its terminal outcome before replay training.
{
    const auto &rules = engine.getState().config;
    Reward reward(rules);

    int startingUpdates = updates;

    while (true)
    {
        float progress = min(1.f, float(updates) / config.decayUpdates);
        float epsilon = 1.f - (1.f - config.minimumEpsilon) * progress;

        bool explore = bernoulli_distribution(epsilon)(generator);
        Agent &behavior = explore ? static_cast<Agent &>(random) : static_cast<Agent &>(greedy);

        auto transition = playLearnerDecision(engine, behavior,
                                              opponents, reward, learner.getFeatureSelection());

        if (!transition)
        {
            break;
        }

        // Replace the shared game loop's shaped reward before inserting into replay.
        transition->reward = 0.f;

        if (transition->done)
        {
            // A trapped learner may still be alive when its decision interval ends.
            int opponentSeat = 3 - rules.P;
            transition->reward = engine.isAlive(rules.P) && !engine.isAlive(opponentSeat) ? 1.f : -1.f;

            cout << "pilot episode=" << episode << " seat=" << rules.P << " opponent=" << opponent
                 << " reward=" << transition->reward << endl;
        }

        if (updates == numeric_limits<int>::max())
        {
            throw overflow_error("Pilot update limit reached");
        }

        learner.remember(*transition);
        learner.train(config.batchSize, config.learningRate);
        updates++;

        if (updates % config.targetEvery == 0)
        {
            learner.updateTargetNetworks();
        }

        if (transition->done)
        {
            break;
        }
    }

    if (startingUpdates == updates)
    {
        throw invalid_argument("Pilot layout has no learner decision");
    }
}  //----- end of playEpisode


static void runPilot(int argc, char **argv)
// Algorithm : Train against seeded opponents and evaluate the greedy learner on the fixed validation set.
{
    if (argc < 6)
    {
        throw invalid_argument("Usage: pilot EPISODES STARTS VALIDATION CONFIG OUTPUT [--resume EPISODE_DIR] [--until-epsilon-floor]");
    }

    int episodes = parseInteger(argv[1], "episodes", 1);
    string settings = readText(argv[4]);
    auto config = readConfig(settings);
    filesystem::path output = argv[5];
    optional<filesystem::path> resume;
    bool untilFloor = false;

    for (int i = 6; i < argc; i++)
    {
        string option = argv[i];
        if (option == "--resume" && i + 1 < argc && !resume)
        {
            resume = argv[++i];
        }
        else if (option == "--until-epsilon-floor")
        {
            untilFloor = true;
        }
        else
        {
            throw invalid_argument("Invalid pilot argument " + option);
        }
    }

    auto params = loadMinimaxParams(config.paramsFile);
    MinimaxAgent tuned(params);

    // The shallow opponent uses the tuned weights with its own search depth.
    params.depth = config.shallowDepth;
    MinimaxAgent shallow(params);

    vector<reference_wrapper<Agent>> opponents{ref(tuned)};
    auto starts = readValidationSet(argv[2], 2, opponents);
    auto validation = readValidationSet(argv[3], 2, opponents);

    // Keep enough information to reject a changed layout set or overwritten parameter file.
    settings += "\n[starts]\n" + readText(argv[2]) + "\n[validation]\n" + readText(argv[3]) +
                "\n[minimax]\n" + readText(config.paramsFile);

#ifdef SNAKE_LIBTORCH
    TorchNetwork::setThreads(config.threads);
#endif

    DQN learner(makeApplicationModelConfig(2), mt19937(config.seed));
    DQNAgent greedy(learner, false);

    int previous = 0;
    int updates = 0;

    if (resume)
    {
        auto state = openSetup((*resume / "pilot.state").string());
        int savedUpdates = 0;
        if (!(state >> previous >> savedUpdates) || previous < 1 || state.get() != '\n' ||
            string(istreambuf_iterator<char>(state), istreambuf_iterator<char>()) != settings)
        {
            throw invalid_argument("Pilot metadata or settings mismatch");
        }

        updates = ModelPersistence{}.loadDQN(learner, (*resume / "model.model").string());
        if (updates != savedUpdates)
        {
            throw invalid_argument("Pilot model and metadata update counts differ");
        }
    }

    if (episodes > numeric_limits<int>::max() - previous)
    {
        throw overflow_error("Pilot episode limit reached");
    }

    filesystem::create_directories(output);
    mt19937 generator;
    RandomLegalAgent random(generator);

    for (int offset = 0; offset < episodes && (!untilFloor || updates < config.decayUpdates); offset++)
    {
        int episode = previous + offset + 1;
        // Recreate the same layout and opponent draws after an episode-boundary resume.
        seed_seq seed{config.seed, episode};
        generator.seed(seed);

        auto &start = starts[uniform_int_distribution<size_t>(0, starts.size() - 1)(generator)];
        auto rules = start.config;
        rules.P = 1 + (episode - 1) % 2;
        GameEngine engine(rules, start.origins);

        // Random and shallow opponents occupy adjacent ranges; tuned gets the remainder.
        int draw = uniform_int_distribution<int>(0, 99)(generator);
        string opponent = "tuned";
        opponents.front() = tuned;
        if (draw < config.randomWeight)
        {
            opponents.front() = random;
            opponent = "random";
        }
        else if (draw < config.randomWeight + config.shallowWeight)
        {
            opponents.front() = shallow;
            opponent = "shallow";
        }

        playEpisode(engine, learner, greedy, random, opponents, config, generator, episode, opponent, updates);

        bool finalEpisode = offset + 1 == episodes || (untilFloor && updates >= config.decayUpdates);
        if (finalEpisode || episode % config.saveEvery == 0)
        {
            saveEpisode(output, episode, updates, settings, learner);
            cout << "checkpoint episode=" << episode << " updates=" << updates << endl;
        }

        if (finalEpisode || episode % config.evaluateEvery == 0)
        {
            auto result = ResultValidation{}.validate(greedy, validation);
            cout << "evaluation episode=" << episode << " wins=" << result.wins << '/' << result.matches
                 << " mean_survival_turns=" << result.meanSurvivalTurns << endl;
        }
    }
}  //----- end of runPilot


//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
int main(int argc, char **argv)
// Algorithm : Run the pilot and report configuration or checkpoint errors to the caller.
{
    try
    {
        runPilot(argc, argv);
    }
    catch (const exception &error)
    {
        cerr << error.what() << endl;
        return 1;
    }

    return 0;
}  //----- end of main
