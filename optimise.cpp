/*************************************************************************
optimise - Implementation of a genetic algorithm to optimise the
parameters of the Minimax algorithm.
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>
#include <chrono>
#include <sstream>
#include <iomanip>

using namespace std;

#include "include/Minimax.h"
#include "include/Structs.h"

// Genetic Algorithm Configuration
const int POPULATION_SIZE = 30;
const int GENERATIONS = 50;
const float MUTATION_RATE = 0.5;
const int TOURNAMENT_SIZE = 3;
const int ELITISM = 2;
const int MATCHES_PER_INDIVIDUAL = 1;

// Game parameters bounds with step size
const vector<int> W_STEPS = {5, 10, 15, 20, 25, 30, 35, 40, 45, 50};
const vector<int> H_STEPS = {5, 10, 15, 20, 25, 30, 35, 40, 45, 50};
const vector<int> M_VALUES = {1, 3, 5, 7, 9, 10};

class Individual {
public:
    MinimaxParams ai1_params;
    MinimaxParams ai2_params;
    double fitness = 0.0;
    int survivalTurns = 0;

    Individual() = default;
    
    Individual(random_device& rd) {
        generate_random_params(rd);
    }

    void generate_random_params(random_device& rd) {
        mt19937 gen(rd());
        auto gen_params = [&]() {
            MinimaxParams p;
            uniform_real_distribution<> space_dist(0, 10);
            uniform_real_distribution<> block_dist(-30, 0);
            uniform_real_distribution<> heat_dist(-5, 0);
            uniform_real_distribution<> length_dist(0, 5);
            uniform_int_distribution<> depth_dist(6, 10);

            p.rewards.selfSpace = space_dist(gen);
            p.rewards.oppSpace = space_dist(gen);
            p.rewards.selfBlockRisk = block_dist(gen);
            p.rewards.oppBlockRisk = block_dist(gen);
            p.rewards.selfHeatRisk = heat_dist(gen);
            p.rewards.oppHeatRisk = heat_dist(gen);
            p.rewards.snakeLength = length_dist(gen);
            p.depth = depth_dist(gen);
            return p;
        };

        ai1_params = gen_params();
        ai2_params = gen_params();
    }

    void save_to_file(const string& filename, const GameConfig& config, bool best) const {
        ofstream file;
        if (best)
        {
            file.open(filename, ios::app);
        }
        else
        {
            file.open(filename);
        }
        file << "[" << config.W << "," << config.H << "," << config.M << "]\n";
        file << fixed << setprecision(5);
        file << "AI1{" << ai1_params.rewards.selfSpace << "," << ai1_params.rewards.oppSpace << ","
             << ai1_params.rewards.selfBlockRisk << "," << ai1_params.rewards.oppBlockRisk << ","
             << ai1_params.rewards.selfHeatRisk << "," << ai1_params.rewards.oppHeatRisk << ","
             << ai1_params.rewards.snakeLength << "}\n";
        file << "AI1Depth=" << ai1_params.depth << "\n";
        file << "AI2{" << ai2_params.rewards.selfSpace << "," << ai2_params.rewards.oppSpace << ","
             << ai2_params.rewards.selfBlockRisk << "," << ai2_params.rewards.oppBlockRisk << ","
             << ai2_params.rewards.selfHeatRisk << "," << ai2_params.rewards.oppHeatRisk << ","
             << ai2_params.rewards.snakeLength << "}\n";
        file << "AI2Depth=" << ai2_params.depth << "\n\n";
    }

    void mutate(random_device& rd) {
        mt19937 gen(rd());
        normal_distribution<> gauss(0.0, 0.2);
        uniform_real_distribution<> prob(0.0, 1.0);

        auto mutate_params = [&](MinimaxParams& params) {
            auto mutate_value = [&](double& value, double min, double max) {
                if(prob(gen) < MUTATION_RATE) {
                    value = clamp(value + gauss(gen), min, max);
                }
            };

            mutate_value(params.rewards.selfSpace, 0, 10);
            mutate_value(params.rewards.oppSpace, 0, 10);
            mutate_value(params.rewards.selfBlockRisk, -30, 0);
            mutate_value(params.rewards.oppBlockRisk, -30, 0);
            mutate_value(params.rewards.selfHeatRisk, -5, 0);
            mutate_value(params.rewards.oppHeatRisk, -5, 0);
            mutate_value(params.rewards.snakeLength, 0, 5);

            if(prob(gen) < MUTATION_RATE) {
                uniform_int_distribution<> depth_dist(1, 10);
                params.depth = depth_dist(gen);
            }
        };

        mutate_params(ai1_params);
        mutate_params(ai2_params);
    }
};

Individual crossover(const Individual& parent1, const Individual& parent2, random_device& rd) {
    mt19937 gen(rd());
    uniform_int_distribution<> coin(0, 1);
    Individual child;

    auto crossover_params = [&](const MinimaxParams& p1, const MinimaxParams& p2) {
        MinimaxParams p;
        p.rewards.selfSpace = coin(gen) ? p1.rewards.selfSpace : p2.rewards.selfSpace;
        p.rewards.oppSpace = coin(gen) ? p1.rewards.oppSpace : p2.rewards.oppSpace;
        p.rewards.selfBlockRisk = coin(gen) ? p1.rewards.selfBlockRisk : p2.rewards.selfBlockRisk;
        p.rewards.oppBlockRisk = coin(gen) ? p1.rewards.oppBlockRisk : p2.rewards.oppBlockRisk;
        p.rewards.selfHeatRisk = coin(gen) ? p1.rewards.selfHeatRisk : p2.rewards.selfHeatRisk;
        p.rewards.oppHeatRisk = coin(gen) ? p1.rewards.oppHeatRisk : p2.rewards.oppHeatRisk;
        p.rewards.snakeLength = coin(gen) ? p1.rewards.snakeLength : p2.rewards.snakeLength;
        p.depth = coin(gen) ? p1.depth : p2.depth;
        return p;
    };

    child.ai1_params = crossover_params(parent1.ai1_params, parent2.ai1_params);
    child.ai2_params = crossover_params(parent1.ai2_params, parent2.ai2_params);

    return child;
}

pair<bool, int> analyze_game_log(const string& filename) {
    ifstream log_file(filename);
    string line;
    int lastTurn = 0;
    int loser = -1;

    while(getline(log_file, line)) {
        if(line.find("**Turn ") != string::npos) {
            lastTurn = stoi(line.substr(7, line.find("**") - 7));
        }
        else if(line.find("is eliminated") != string::npos) {
            loser = (line.find("AI 0") != string::npos) ? 0 : 1;
        }
    }

    return {loser != 0, lastTurn}; // Returns (ai1_won, survivalTurns)
}

double evaluate_individual(Individual& ind, const GameConfig& config, random_device& rd) {
    double total_fitness = 0.0;
    
    for(int i = 0; i < MATCHES_PER_INDIVIDUAL; ++i) {
        // Save parameters
        ind.save_to_file("current_params.txt", config, false);

        // Build command
        stringstream cmd;
        cmd << "python3 snake.py -g " << config.W << " " << config.H 
            << " -G " << config.M << " bot bot > game_log.txt 2>&1";

        // Execute
        int result = system(cmd.str().c_str());
        
        // Analyze results
        auto [ai1_won, turns] = analyze_game_log("game_log.txt");
        ind.survivalTurns += turns;

        // Print match result
        cout << "Game " << config.W << "x" << config.H << " M" << config.M 
             << " - " << (ai1_won ? "AI1" : "AI2") << " won after " 
             << turns << " turns\n";

        // Update fitness (25% for winning, 75% for survival time)
        total_fitness += (ai1_won ? 0.25 : 0) + 0.75*((float)turns / (float)(config.W * config.H * config.M));
    }
    
    ind.fitness = total_fitness / MATCHES_PER_INDIVIDUAL;
    return ind.fitness;
}

void run_genetic_algorithm(const GameConfig& config) {
    random_device rd;
    mt19937 gen(rd());
    
    vector<Individual> population(POPULATION_SIZE);
    generate(population.begin(), population.end(), [&]() { return Individual(rd); });

    Individual best_individual;
    double best_fitness = -1;

    for(int generation = 0; generation < GENERATIONS; ++generation) {
        // Evaluate population
        for(auto& ind : population) {
            evaluate_individual(ind, config, rd);
            if(ind.fitness > best_fitness) {
                best_fitness = ind.fitness;
                best_individual = ind;
            }
        }

        // Sort and create new population
        sort(population.begin(), population.end(), 
            [](const Individual& a, const Individual& b) { return a.fitness > b.fitness; });

        vector<Individual> new_population;
        for(int i = 0; i < ELITISM; ++i) {
            new_population.push_back(population[i]);
        }

        uniform_int_distribution<size_t> dist(0, POPULATION_SIZE-1);
        while(new_population.size() < POPULATION_SIZE) {
            const auto& parent1 = population[dist(gen)];
            const auto& parent2 = population[dist(gen)];
            new_population.push_back(crossover(parent1, parent2, rd));
        }

        population = new_population;

        cout << "Config [" << config.W << "," << config.H << "," << config.M 
             << "] Generation " << generation << " completed. Best fitness: " 
             << best_fitness << endl;
    }

    // Save best for this configuration
    best_individual.save_to_file("best_params.txt", config, true);
}

int main() {
    // Generate all configurations
    vector<GameConfig> configs;
    for(int W : W_STEPS) {
        for(int H : H_STEPS) {
            for(int M : M_VALUES) {
                configs.push_back({W, H, M});
            }
        }
    }

    // Process each configuration
    for(const auto& config : configs) {
        cout << "\n=== Starting configuration [" << config.W << "," 
             << config.H << "," << config.M << "] ===\n";
        run_genetic_algorithm(config);
    }

    return 0;
}