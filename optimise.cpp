/*************************************************************************
optimise - Tunes Minimax parameters on supplied validation matches
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//------ Implementation of the module <optimise> (file optimise.cpp) ------

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace std;

//------------------------------------------------------- Personal Include
#include "src/Agent/MinimaxAgent.h"
#include "src/DQN/ResultValidation.h"
#include "src/GameEngine/InputHandler.h"
#include "src/Minimax/MinimaxParamsFile.h"

//------------------------------------------------------------------ Types
struct OptimiseConfig {
    string validationSet;
    string output;
    int populationSize = 30;
    int generations = 50;
    int seed = mt19937::default_seed;
};

struct Individual {
    MinimaxParams params;
    ValidationResult result;
    float score = -1.f;
};

struct WeightRange {
    double Rewards::*weight;
    double minimum;
    double maximum;
};

//-------------------------------------------------------------- Constants
static constexpr int ELITISM = 2;
static constexpr WeightRange WEIGHT_RANGES[] = {
    {&Rewards::selfSpace, 0, 10},
    {&Rewards::oppSpace, 0, 10},
    {&Rewards::selfBlockRisk, -30, 0},
    {&Rewards::oppBlockRisk, -30, 0},
    {&Rewards::selfHeatRisk, -5, 0},
    {&Rewards::oppHeatRisk, -5, 0},
    {&Rewards::snakeLength, 0, 5}
};

//---------------------------------------------------------------- PRIVATE
static OptimiseConfig readArguments(int argc, char **argv)
// Algorithm : Read the validation and output paths and optional genetic search settings.
{
    if (argc < 3)
    {
        throw invalid_argument("Usage: optimise VALIDATION_SET OUTPUT [--population N] [--generations N] [--seed N]");
    }

    OptimiseConfig config;
    config.validationSet = argv[1];
    config.output = argv[2];
    for (int index = 3; index < argc; index += 2)
    {
        string option = argv[index];
        if (index + 1 == argc)
        {
            throw invalid_argument("Missing value for " + option);
        }

        string argument = argv[index + 1];
        size_t end = 0;
        int value = stoi(argument, &end);
        if (end != argument.size() || value < 0)
        {
            throw invalid_argument("Invalid value for " + option);
        }

        if (option == "--population")
        {
            config.populationSize = value;
        }
        else if (option == "--generations")
        {
            config.generations = value;
        }
        else if (option == "--seed")
        {
            config.seed = value;
        }
        else
        {
            throw invalid_argument("Unknown option " + option);
        }
    }

    if (config.populationSize <= ELITISM || config.generations < 1)
    {
        throw invalid_argument("Expected at least three individuals and one generation");
    }

    return config;
}  //----- end of readArguments


static vector<ValidationMatch> readValidationSet(const string &filename, Agent &opponent)
// Algorithm : Load complete competition setups and use the fixed opponent in every non-candidate seat.
{
    ifstream input(filename);
    if (!input)
    {
        throw runtime_error("Cannot open validation set " + filename);
    }

    vector<ValidationMatch> matches;
    while ((input >> ws).peek() != char_traits<char>::eof())
    {
        GameConfig rules = InputHandler::readGameConfig(input);
        if (rules.N < 2)
        {
            throw invalid_argument("Optimization matches require at least two players");
        }

        auto origins = InputHandler::readPlayersOrigins(rules.N, input);
        GameEngine{rules, origins};
        vector<reference_wrapper<Agent>> opponents(rules.N - 1, ref(opponent));
        matches.push_back({rules, move(origins), move(opponents)});
    }

    if (input.bad() || !input.eof() || matches.empty())
    {
        throw invalid_argument("Expected at least one complete validation setup in " + filename);
    }

    return matches;
}  //----- end of readValidationSet


static MinimaxParams randomParams(mt19937 &generator)
// Algorithm : Sample each reward independently in its legacy range and start with depth six through ten.
{
    MinimaxParams params;
    for (const auto &range : WEIGHT_RANGES)
    {
        params.rewards.*(range.weight) = uniform_real_distribution<double>(range.minimum, range.maximum)(generator);
    }

    params.depth = uniform_int_distribution<int>(6, 10)(generator);
    return params;
}  //----- end of randomParams


static MinimaxParams crossover(const MinimaxParams &first, const MinimaxParams &second, mt19937 &generator)
// Algorithm : Choose each reward and the depth independently from one of the two parents.
{
    MinimaxParams child;
    uniform_int_distribution<int> coin(0, 1);
    for (const auto &range : WEIGHT_RANGES)
    {
        const MinimaxParams &parent = coin(generator) ? first : second;
        child.rewards.*(range.weight) = parent.rewards.*(range.weight);
    }

    child.depth = coin(generator) ? first.depth : second.depth;
    return child;
}  //----- end of crossover


static Individual optimise(const OptimiseConfig &config, const vector<ValidationMatch> &matches)
// Algorithm : Evaluate each generation on the same matches, retain two elites and cross uniformly sampled parents.
{
    mt19937 generator(config.seed);
    vector<Individual> population(config.populationSize);
    for (auto &individual : population)
    {
        individual.params = randomParams(generator);
    }

    ResultValidation validation;
    Individual best;
    uniform_int_distribution<int> parentIndex(0, config.populationSize - 1);

    for (int generation = 0; generation < config.generations; generation++)
    {
        for (auto &individual : population)
        {
            MinimaxAgent agent(individual.params);
            individual.result = validation.validate(agent, matches);
            individual.score = validation.calculateScore(individual.result);
            if (individual.score > best.score)
            {
                best = individual;
            }
        }

        if (generation + 1 == config.generations)
        {
            break;
        }

        // Keep population order when candidates have equal win rates.
        stable_sort(population.begin(), population.end(), [](const Individual &first, const Individual &second)
        {
            return first.score > second.score;
        });

        vector<Individual> next(population.begin(), population.begin() + ELITISM);
        while (next.size() < population.size())
        {
            const auto &first = population[parentIndex(generator)].params;
            const auto &second = population[parentIndex(generator)].params;
            Individual child;
            child.params = crossover(first, second, generator);
            next.push_back(child);
        }

        population.swap(next);
    }

    return best;
}  //----- end of optimise


//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
int main(int argc, char **argv)
// Algorithm : Optimize against the default Minimax opponent, save the winner and report validation metrics.
{
    try
    {
        OptimiseConfig config = readArguments(argc, argv);
        MinimaxAgent opponent(MinimaxParams{});
        auto matches = readValidationSet(config.validationSet, opponent);
        Individual best = optimise(config, matches);
        saveMinimaxParams(best.params, config.output);
        cout << "optimisation wins=" << best.result.wins << '/' << best.result.matches
             << " win_rate=" << best.score << " mean_survival_turns=" << best.result.meanSurvivalTurns
             << " depth=" << best.params.depth << " output=" << config.output << '\n';
    }
    catch (const exception &error)
    {
        cerr << error.what() << endl;
        return 1;
    }

    return 0;
}  //----- end of main
