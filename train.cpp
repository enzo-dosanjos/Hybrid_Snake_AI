/*************************************************************************
train - Runs and resumes neural Snake training
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//--------- Implementation of the module <train> (file train.cpp) ---------

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <cmath>
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
#include "src/Agent/MinimaxAgent.h"
#include "src/ApplicationModel.h"
#include "src/DQN/ModelPersistence.h"
#include "src/DQN/ResultValidation.h"
#include "src/GameEngine/InputHandler.h"
#include "src/TrainingLoop.h"

//------------------------------------------------------------------ Types
struct TrainingConfig {
    int episodes = 0;
    string trainingSetup;
    string validationSet;
    string checkpoint;
    bool resume = false;
    optional<float> hybridWeight;
    int batchSize = 32;
    float learningRate = 0.00001f;
    int targetEvery = 100;
    int saveEvery = 10;
    int evaluateEvery = 10;
    int seed = mt19937::default_seed;
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


static TrainingConfig readArguments(int argc, char **argv)
// Algorithm : Parse the required paths and episode count followed by optional training settings.
{
    if (argc < 5)
    {
        throw invalid_argument(
            "Usage: train EPISODES TRAIN_SETUP VALIDATION_SET CHECKPOINT [--resume] [--hybrid LAMBDA] "
            "[--batch-size N] [--learning-rate RATE] [--target-every UPDATES] "
            "[--save-every EPISODES] [--evaluate-every EPISODES] [--seed N]"
        );
    }

    TrainingConfig config;
    config.episodes = parseInteger(argv[1], "EPISODES", 1);
    config.trainingSetup = argv[2];
    config.validationSet = argv[3];
    config.checkpoint = argv[4];

    for (int index = 5; index < argc; index++)
    {
        string option = argv[index];
        if (option == "--resume")
        {
            config.resume = true;
            continue;
        }

        if (++index == argc)
        {
            throw invalid_argument("Missing value for " + option);
        }

        string value = argv[index];
        if (option == "--hybrid")
        {
            config.hybridWeight = parseHybridWeight(value);
        }
        else if (option == "--batch-size")
        {
            config.batchSize = parseInteger(value, option, 1);
        }
        else if (option == "--learning-rate")
        {
            size_t end = 0;
            config.learningRate = stof(value, &end);
            if (end != value.size() || !isfinite(config.learningRate) || config.learningRate <= 0.f)
            {
                throw invalid_argument("Learning rate must be finite and positive");
            }
        }
        else if (option == "--target-every")
        {
            config.targetEvery = parseInteger(value, option, 1);
        }
        else if (option == "--save-every")
        {
            config.saveEvery = parseInteger(value, option, 1);
        }
        else if (option == "--evaluate-every")
        {
            config.evaluateEvery = parseInteger(value, option, 1);
        }
        else if (option == "--seed")
        {
            config.seed = parseInteger(value, option, 0);
        }
        else
        {
            throw invalid_argument("Unknown option " + option);
        }
    }

    return config;
}  //----- end of readArguments


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
        GameEngine(rules, origins);
        matches.push_back({rules, move(origins), opponents});
    }

    if (input.bad() || !input.eof() || matches.empty())
    {
        throw invalid_argument("Expected at least one complete validation setup in " + filename);
    }

    return matches;
}  //----- end of readValidationSet


static void runTraining(const TrainingConfig &config)
// Algorithm : Replay a configured matchup, update the learner after each decision, and save and evaluate between episodes.
{
    ifstream input = openSetup(config.trainingSetup);
    GameConfig rules = InputHandler::readGameConfig(input);
    auto origins = InputHandler::readPlayersOrigins(rules.N, input);
    if (rules.N < 2 || (input >> ws).peek() != char_traits<char>::eof() || input.bad() || !input.eof())
    {
        throw invalid_argument("Expected exactly one training setup with at least two players");
    }

    GameEngine initial(rules, origins);
    Reward reward(rules);
    MinimaxAgent minimax(MinimaxParams{});
    vector<reference_wrapper<Agent>> opponents(rules.N - 1, ref(minimax));
    auto validationSet = readValidationSet(config.validationSet, rules.N, opponents);

    if (!config.resume && filesystem::exists(config.checkpoint))
    {
        throw invalid_argument("Checkpoint already exists. Use --resume or provide a new path.");
    }

    DQN learner(makeApplicationModelConfig(rules.N), mt19937(config.seed));
    ModelPersistence persistence;
    ResultValidation validation;
    int updates = config.resume ? persistence.loadDQN(learner, config.checkpoint) : 0;
    auto behavior = makeNeuralAgent(learner, config.hybridWeight, true);
    auto evaluated = makeNeuralAgent(learner, config.hybridWeight, false);

    for (int episode = 0; episode < config.episodes; episode++)
    {
        GameEngine engine = initial;
        int startingUpdates = updates;

        while (auto transition = playLearnerDecision(engine, *behavior, opponents, reward,
                                                     learner.getFeatureSelection()))
        {
            if (updates == numeric_limits<int>::max())
            {
                throw overflow_error("Training step limit reached");
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

        if (updates == startingUpdates)
        {
            throw invalid_argument("Training setup ends before the learner can make a decision");
        }

        int completed = episode + 1;
        if (completed % config.saveEvery == 0 || completed == config.episodes)
        {
            persistence.saveDQN(learner, config.checkpoint, updates);
            cout << "checkpoint episode=" << completed << " updates=" << updates << endl;
        }

        if (completed % config.evaluateEvery == 0 || completed == config.episodes)
        {
            auto result = validation.validate(*evaluated, validationSet);
            cout << "evaluation episode=" << completed << " updates=" << updates
                 << " wins=" << result.wins << '/' << result.matches
                 << " win_rate=" << validation.calculateScore(result)
                 << " mean_survival_turns=" << result.meanSurvivalTurns << endl;
        }
    }
}  //----- end of runTraining


//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
int main(int argc, char **argv)
// Algorithm : Run the requested training workflow and report input and checkpoint errors on stderr.
{
    try
    {
        runTraining(readArguments(argc, argv));
    }
    catch (const exception &error)
    {
        cerr << error.what() << endl;
        return 1;
    }

    return 0;
}  //----- end of main
