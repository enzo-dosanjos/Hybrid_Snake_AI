/*************************************************************************
NNAI - todo
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//--------- Implementation of the module <NNAI> (file NNAI.cpp) ---------

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <iostream>
#include <algorithm>
#include <fstream>
#include <iterator>

using namespace std;

//------------------------------------------------------- Personal Include
#include "../include/NNAI.h"

//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
void NNAI::copyWeights (const NNAI &src) {
    // copy weights and biases of the FCNN and the CNN
    aiFCNN.copyWeights(src.aiFCNN);

    aiCNN.copyWeights(src.aiCNN);
}


string NNAI::selectAction(const vector<float> &boardState, const vector<float> &extraState)
// Algorithm : Selects an action using epsilon-greedy policy
{
    // Random number generator
    uniform_real_distribution<float> dist(0.0, 1.0);

    if (dist(gen) < epsilon)
    {
        // Exploration: choose a random action
        uniform_int_distribution<int> actionDist(0, LABEL_ACTIONS.size() - 1);
        int randInd = actionDist(gen);
        return LABEL_ACTIONS[randInd];
    }
    else
    {
        // Exploitation: choose the action with the highest Q-value
        vector<float> qValues = forwardPropagation(boardState, extraState); // Compute Q-values
        auto itBestVal = max_element(qValues.begin(), qValues.end());
        int bestInd = distance(qValues.begin(), itBestVal);  // Use distance to get the index of the max element

        return LABEL_ACTIONS[bestInd];
    }
}  //----- End of selectAction


vector<float> NNAI::forwardPropagation(const vector<float> &boardState, const vector<float> &extraState)
// Algorithm : Forward propagation through the neural network (CNN + FCNN)
{
    // Reshape the board state to match the CNN input format
    Matrix<vector<float>> reshapedBoardState = helper.reshapeBoardState(boardState);

    // Forward pass through CNN
    Matrix<vector<float>> cnnOutput = aiCNN.forwardPropagation(reshapedBoardState);

    // Global average pooling to get a flat vector
    vector<float> pooled = aiCNN.globalAvgPooling(cnnOutput);

    // Combine with extra features
    vector<float> combined = combineFeatures(pooled, extraState);

    // Forward propagation through FCNN
    return aiFCNN.forwardPropagation(combined);
}  //----- End of forwardPropagation


float NNAI::computeReward(const vector<float> &boardState, const vector<float> &extraState,
                          const vector<float> &prevExtraState, const PlayersData &playerState,
                          int myPlayer, int nbOppKillSetup, bool oppWillDie, bool oppDied, int turn)
// Algorithm : Compute the reward for a given state
{
    // - If my snake's head collides with a border or its own body, return negative lot
    // - If the opponent's snake head collides, return a positive lot
    // - Otherwise, return a small positive reward for survival
    float score = 0.0;

    bool terminal = helper.checkTerminalState(boardState, playerState);
    bool playerAlive = !helper.checkDeath(playerState, boardState, myPlayer);

    float maxTurns = (M > 1) ? 0.8*(W*H*M) : (W*H*M);
    NNRewards rewards(W, H, turn, maxTurns);

    if (playerAlive && terminal)
    {
        score += rewards.winningMove;
    }
    else if (terminal)
    {
        return rewards.losingMove;
    }
    else
    {
        for (const auto &player : playerState)
        {
            if (player.first != myPlayer)
            {
                if (helper.checkDeath(playerState, boardState, player.first))
                {
                    cout << "> everyone dead" << endl;
                    score += rewards.winningMove;
                }
            }
        }
    }

    if (oppWillDie || oppDied)  // todo: check if it's my ai that killed the opp (works in 1v1)
    {
        score += rewards.winningMove;
    }

    score += rewards.killingSetupMove * nbOppKillSetup;

    const int nbOpp = (playerState.size()-1);
    const int ESCAPE_ROUTES_INDEX = 6*nbOpp + 3;
    const int KILL_ZONE_INDEX = ESCAPE_ROUTES_INDEX + nbOpp;
    const int PHASE_INDEX = KILL_ZONE_INDEX + nbOpp;
    const int ENTROPY_INDEX = PHASE_INDEX + 1;

    // Aggression rewards
    float aggressionScore = 0.0;

    for (int i = 0; i < playerState.size()-1; i++)
    {
        float killZoneValue = 1.0 / (1.0 + exp(-4.0*(extraState[KILL_ZONE_INDEX + i]-0.3)));  // sigmoid scaling for the killzone value
        aggressionScore += rewards.killZoneEnterMove * killZoneValue;
    }

    for (int i = 0; i < playerState.size()-1; i++)
    {
        if(prevExtraState[ESCAPE_ROUTES_INDEX + i] - extraState[ESCAPE_ROUTES_INDEX + i] > 0)
        {
            aggressionScore += rewards.escapeReductionMove * (prevExtraState[ESCAPE_ROUTES_INDEX + i] - extraState[ESCAPE_ROUTES_INDEX + i]);
        }
    }

    // Entropy penalty
    const float minAcceptableEntropy = 0.7; // ~3 moves used
    if(extraState[ENTROPY_INDEX] < minAcceptableEntropy) {
        aggressionScore += ((minAcceptableEntropy - extraState[ENTROPY_INDEX])/minAcceptableEntropy) * rewards.entropyPenalty;
    }

    // Depends on the size for exponential reward
    score += rewards.survivingMove * playerState.at(myPlayer).first.size();

    // Z-score normalization
    static float reward_mean = 0.0f;
    static float reward_std = 1.0;
    float z_score = ((score + aggressionScore) - reward_mean) / reward_std;
    reward_mean = 0.99f * reward_mean + 0.01f * (score + aggressionScore);
    reward_std = 0.99f * reward_std + 0.01f * fabs((score + aggressionScore) - reward_mean);
    return z_score;
}  //----- End of computeReward


void NNAI::trainNN (vector<Transition> &batch, NNAI &targetNetwork, float learningRate, float gamma) {
    // Reset gradients for both networks
    aiCNN.resetGradients();
    aiFCNN.resetGradients();

    for (auto &transition : batch)
    {
        // Forward propagation on the current state
        vector<float> qValues = forwardPropagation(transition.boardState, transition.extraState);

        // Forward propagation on the next state with the target network
        vector<float> targetQValues =
            targetNetwork.forwardPropagation(transition.nextBoardState, transition.nextExtraState);

        // Compute the error for FCNN
        aiFCNN.computeError(targetQValues, transition, gamma);

        // Backpropagation through FCNN
        aiFCNN.backPropagation();

        // Spread error to CNN
        aiCNN.spreadFCNNError(aiFCNN.getError(0));

        // Backpropagation through CNN
        aiCNN.backPropagation();

        // Accumulate gradients
        aiFCNN.accumulateGradient();
        aiCNN.accumulateGradient();
    }

    // Update weights
    aiFCNN.updateWeights(learningRate);
    aiCNN.updateWeights(learningRate);

    // Decay exploration rate
    decayEpsilon();
} //----- end of trainNN


ValidationResult NNAI::analyzeResults(const string &filename)
// Algorithm : Analyze the results of a game and return a ValidationResult
// with extracted metrics from the game log file
{
    ifstream file(filename);
    if(!file) {
        cerr << "Error opening file for reading: " << filename << endl;
        return {};
    }

    ValidationResult result;
    result.victory = false;
    result.survivalTurns = 0;
    result.avgReward = 0.0;
    result.maxReward = -numeric_limits<float>::max();
    result.minReward = numeric_limits<float>::max();
    result.actionEntropy = 0.0;
    result.qValueConsistency = 0.0;
    result.deathPenalty = 0.0;

    map<string, int> action_counts = {{"left", 0}, {"right", 0}, {"up", 0}, {"down", 0}};
    int total_actions = 0;
    vector<float> rewards;
    vector<float> q_values;
    string line;

    while(getline(file, line))
    {
        // Detect win
        if(line.find("is eliminated") != string::npos)
        {
            result.victory = (line.find("botOpp") != string::npos);
        }

        // Count survival turns
        if(line.find("**Turn ") != string::npos)
        {
            result.survivalTurns++;
        }

        // Analyse rewards
        if(line.find("> reward: ") != string::npos)
        {
            size_t pos = line.find("> reward: ");
            if(pos != string::npos && pos + 10 < line.size())
            {
                try
                {
                    float reward = stof(line.substr(pos + 10));
                    rewards.push_back(reward);
                    result.maxReward = max(result.maxReward, reward);
                    result.minReward = min(result.minReward, reward);
                }
                catch(const exception& e)
                {
                    cerr << "Error parsing reward in line: " << line << endl;
                    cerr << "Exception: " << e.what() << endl;
                }
            }
        }

        // Analyse Q-values
        if(line.find("DQN Q-Value: ") != string::npos)
        {
            size_t pos = line.find("DQN Q-Value: ");
            try
            {
                float q_val = stof(line.substr(pos + 13));
                q_values.push_back(q_val);
            }
            catch(const exception& e)
            {
                cerr << "Error parsing Q-value in line: " << line << endl;
                cerr << "Exception: " << e.what() << endl;
            }
        }

        // Count actions
        if(line.find("'s move : ") != string::npos && line.find("botDQN") != string::npos)
        {
            size_t pos = line.find(": ") + 2;
            string action = line.substr(pos);
            action_counts[action]++;
            total_actions++;
        }

        // Detect death penalty
        if(line.find("> done: 1") != string::npos && line.find("reward: "))
        {
            size_t pos = line.find("reward: ");
            try
            {
                result.deathPenalty = stof(line.substr(pos + 7));
            }
            catch(const exception& e)
            {
                cerr << "Error parsing death penalty in line: " << line << endl;
                cerr << "Exception: " << e.what() << endl;
            }
        }
    }

    // Compute average reward during the game
    if(!rewards.empty())
    {
        float sum = accumulate(rewards.begin(), rewards.end(), 0.0f);
        result.avgReward = sum / rewards.size();
    }

    // Compute action entropy
    float entropy = 0.0;
    for(const auto& [action, count] : action_counts)
    {
        if(count > 0) {
            float prob = static_cast<float>(count) / total_actions;
            entropy -= prob * log2(prob);
        }
    }
    result.actionEntropy = entropy;

    // Compute Q-value consistency
    if(!q_values.empty()) {
        float q_mean = accumulate(q_values.begin(), q_values.end(), 0.0f) / q_values.size();
        vector<float> diff(q_values.size());
        transform(q_values.begin(), q_values.end(), diff.begin(), [q_mean](float x) { return pow(x - q_mean, 2); });
        result.qValueConsistency = sqrt(accumulate(diff.begin(), diff.end(), 0.0f) / q_values.size());
    }

    result.actionCounts = action_counts;

    return result;
}  //----- End of analyzeResults



float NNAI::calculateScore(const ValidationResult &result)
// Algorithm : Calculate a composite score from the validation results
{
    const float VICTORY_WEIGHT = 2.0;
    const float SURVIVAL_WEIGHT = 0.5;
    const float REWARD_WEIGHT = 1.0;
    const float ENTROPY_WEIGHT = 0.7;
    const float CONSISTENCY_WEIGHT = 0.3;

    float score = 0.0;

    // Victory
    score += result.victory ? VICTORY_WEIGHT : 0.0;

    // Survival turns (normalized)
    score += (result.survivalTurns / 1000.0) * SURVIVAL_WEIGHT;

    // Average reward (normalized)
    score += (result.avgReward / 1000.0) * REWARD_WEIGHT;

    // Action entropy (max 2 bits for 4 actions)
    score += (result.actionEntropy / 2.0) * ENTROPY_WEIGHT;

    // Q-value consistency (inversely proportional)
    score += (1.0 / (1.0 + result.qValueConsistency)) * CONSISTENCY_WEIGHT;

    return score;
}  //----- End of calculateScore


void NNAI::validate(const vector<GameConfig>& validationSet)
// Algorithm : Validate the agent on a set of configurations and save the scores
{
    vector<float> scores;

    for(const auto& config : validationSet) {
        float total_score = 0.0;

        for(int i = 0; i < 2; ++i) { // 4 games per configuration
            // Play the game as the first player
            string log_file = "validation_" + to_string(time(nullptr)) + ".log";
            string cmd = "python3 snake.py -g " + to_string(config.W) + " " + to_string(config.H)
                       + " -G " + to_string(config.M) + " botDQN botOpp >> " + log_file + " 2>&1";
            (void)system(cmd.c_str());

            // Analyse results
            ValidationResult result = analyzeResults(log_file);
            cout << "> ------------ Results --------------" << endl;
            cout << "> Victory: " << boolalpha << result.victory << endl;
            cout << "> Survived turns: " << result.survivalTurns << endl;
            cout << "> Average reward: " << result.avgReward << endl;
            cout << "> Actions' entropy: " << result.actionEntropy << endl;
            cout << "> Composite score: " << calculateScore(result) << endl << endl;

            total_score += calculateScore(result);
        }

        for(int i = 0; i < 2; ++i) {
            // Play the game as the second player
            string log_file = "validation_" + to_string(time(nullptr)) + ".log";
            string cmd = "python3 snake.py -g " + to_string(config.W) + " " + to_string(config.H)
                       + " -G " + to_string(config.M) + " botOpp botDQN >> " + log_file + " 2>&1";
            (void)system(cmd.c_str());

            // Analyse results
            ValidationResult result = analyzeResults(log_file);
            cout << "> ------------ Results --------------" << endl;
            cout << "> Victory: " << boolalpha << result.victory << endl;
            cout << "> Survived turns: " << result.survivalTurns << endl;
            cout << "> Average reward: " << result.avgReward << endl;
            cout << "> Actions' entropy: " << result.actionEntropy << endl;
            cout << "> Composite score: " << calculateScore(result) << endl << endl;

            total_score += calculateScore(result);
        }

        scores.push_back(total_score / 10.0);
    }

    // Save the scores
    ofstream("validation_scores.txt", ios::app) << "[" << time(nullptr) << "] ";
    copy(scores.begin(), scores.end(), ostream_iterator<float>(cout, " "));
}  //----- End of validate


void NNAI::saveToFile(const string& filename, int trainingStep)
// Algorithm : Save the model and replay buffer to files
{
    // Save model to one file
    string modelFile = filename + ".model";
    saveModelToFile(modelFile);

    // Save replay buffer to a separate file
    string bufferFile = filename + ".buffer";
    saveReplayBufferToFile(bufferFile, trainingStep);
}  //----- End of saveToFile


void NNAI::loadFromFile(const string& filename, int &trainingStep)
// Algorithm : Load the model and replay buffer from files
{
    // Load model from one file
    string modelFile = filename + ".model";
    loadModelFromFile(modelFile);

    // Load replay buffer from a separate file
    string bufferFile = filename + ".buffer";
    loadReplayBufferFromFile(bufferFile, trainingStep);
}  //----- End of loadFromFile


void NNAI::saveModelToFile(const string& filename)
// Algorithm : Save the FCNN and CNN architecture and layers
{
    ofstream file(filename, ios::binary);
    if(!file)
    {
        cout << "> Cannot open file for writing" << endl;
    }

    // Save the FCNN architecture
    int numFCNNLayers = aiFCNN.getNNSize();
    file.write(reinterpret_cast<char*>(&numFCNNLayers), sizeof(numFCNNLayers));

    for(const auto &layer : aiFCNN.getNN())
    {
        // Save the layer's parameters
        saveLayer(file, layer);
    }

    // Save the CNN architecture
    int numCNNLayers = aiCNN.getNNSize();
    file.write(reinterpret_cast<char*>(&numCNNLayers), sizeof(numCNNLayers));

    // save the weigths and biases of the CNN
    for(const auto &layer : aiCNN.getNN())
    {
        saveConvLayer(file, layer);
    }
}  //----- End of saveModelToFile



void NNAI::saveReplayBufferToFile(const string& filename, int trainingStep)
// Save every transition of the replayBuffer and the training parameters
{
    ofstream file(filename, ios::binary);
    if(!file) {
        cout << "> Cannot open file for writing: " << filename << endl;
        return;
    }

    // Save buffer size
    int bufferSize = replayBuffer.size();
    file.write(reinterpret_cast<const char*>(&bufferSize), sizeof(bufferSize));

    // Save each transition
    for(const auto& transition : replayBuffer.getTransitions()) {
        // Save reward, done flag and tdError
        file.write(reinterpret_cast<const char*>(&transition.reward), sizeof(float));
        file.write(reinterpret_cast<const char*>(&transition.done), sizeof(bool));
        file.write(reinterpret_cast<const char*>(&transition.tdError), sizeof(float));

        // Save state vectors
        saveVector(file, transition.boardState);
        saveVector(file, transition.extraState);
        saveVector(file, transition.nextBoardState);
        saveVector(file, transition.nextExtraState);

        // Save action
        saveString(file, transition.action);
    }

    // Training parameters
    file.write(reinterpret_cast<char*>(&trainingStep), sizeof(trainingStep));
    file.write(reinterpret_cast<char*>(&epsilon), sizeof(epsilon));

    file.close();
}  //----- End of saveReplayBufferToFile



void NNAI::loadModelFromFile(const string& filename)
// Algorithm : Load the FCNN and CNN architecture and layers
{
    ifstream file(filename, ios::binary);
    if(!file)
    {
        cout << "> Cannot open file for reading" << endl;
        return;
    }

    // Load the FCNN architecture
    int numLayers;
    file.read(reinterpret_cast<char*>(&numLayers), sizeof(numLayers));

    aiFCNN.clearNN();

    // Load each layer
    for(int i = 0; i < numLayers; i++)
    {
        loadLayers(file);
    }


    // Load CNN architecture
    int numCNNLayers;
    file.read(reinterpret_cast<char*>(&numCNNLayers), sizeof(numCNNLayers));

    aiCNN.clearNN();

    // Load each CNN layer
    for(int i = 0; i < numCNNLayers; i++)
    {
        loadConvLayers(file);
    }

    file.close();
}  //----- End of loadModelFromFile


void NNAI::loadReplayBufferFromFile(const string& filename, int &trainingStep)
// Algorithm : Load the replay buffer and training parameters from a file
{
    ifstream file(filename, ios::binary);
    if(!file) {
        cout << "> Cannot open file for reading: " << filename << endl;
        return;
    }

    // Clear existing buffer
    replayBuffer.clearReplayBuffer();

    // Load buffer size
    int bufferSize;
    file.read(reinterpret_cast<char*>(&bufferSize), sizeof(bufferSize));

    // Load each transition
    for(int i = 0; i < bufferSize; i++) {
        Transition t;

        // Load reward and done flag
        file.read(reinterpret_cast<char*>(&t.reward), sizeof(float));
        file.read(reinterpret_cast<char*>(&t.done), sizeof(bool));
        file.read(reinterpret_cast<char*>(&t.tdError), sizeof(float));

        // Load state vectors
        loadVector(file, t.boardState);
        loadVector(file, t.extraState);
        loadVector(file, t.nextBoardState);
        loadVector(file, t.nextExtraState);

        // Load action
        loadString(file, t.action);

        // Add to buffer
        replayBuffer.push(t);
    }

    // Load training parameters
    file.read(reinterpret_cast<char*>(&trainingStep), sizeof(trainingStep));
    file.read(reinterpret_cast<char*>(&epsilon), sizeof(epsilon));

    file.close();
}  //----- End of loadReplayBufferFromFile


//-------------------------------------------------------------- PROTECTED
void NNAI::saveLayer(ofstream &file, const FullyConnectedLayer &layer)
// Algorithm : Save the parameters of a fully connected layer to a file
{
    // Save layer type
    saveString(file, layer.type);

    // Save dimensions
    file.write(reinterpret_cast<const char*>(&layer.inputSize), sizeof(int));
    file.write(reinterpret_cast<const char*>(&layer.outputSize), sizeof(int));

    // Save weights
    for(const auto &row : layer.weights) {
        for(float w : row) {
            file.write(reinterpret_cast<const char*>(&w), sizeof(float));
        }
    }

    // Save biases
    for(float b : layer.biases) {
        file.write(reinterpret_cast<const char*>(&b), sizeof(float));
    }
}  //----- End of saveLayer


void NNAI::saveConvLayer(ofstream &file, const ConvolutionalLayer &layer)
// Algorithm : Save the parameters of a convolutional layer to a file
{
    // Save layer type
    saveString(file, layer.type);

    // Save dimensions and parameters
    file.write(reinterpret_cast<const char*>(&layer.kernelSize), sizeof(int));
    file.write(reinterpret_cast<const char*>(&layer.stride), sizeof(int));
    file.write(reinterpret_cast<const char*>(&layer.padding), sizeof(int));
    file.write(reinterpret_cast<const char*>(&layer.outputChannels), sizeof(int));
    file.write(reinterpret_cast<const char*>(&layer.inputChannels), sizeof(int));

    // Save weights
    for(int ic = 0; ic < layer.inputChannels; ic++) {
        for(int oc = 0; oc < layer.outputChannels; oc++) {
            for(int kh = 0; kh < layer.kernelSize; kh++) {
                for(int kw = 0; kw < layer.kernelSize; kw++) {
                    file.write(reinterpret_cast<const char*>(&layer.weights[ic][oc][kh][kw]), sizeof(float));
                }
            }
        }
    }

    // Save biases
    for(float b : layer.biases) {
        file.write(reinterpret_cast<const char*>(&b), sizeof(float));
    }
}  //----- End of saveConvLayer


void NNAI::saveVector(ofstream& file, const vector<float>& vec)
// Algorithm : Save a vector of floats to a file
{
    int size = vec.size();
    file.write(reinterpret_cast<const char*>(&size), sizeof(int));

    if(size > 0) {
        file.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(float));
    }
}  //----- End of saveVector


void NNAI::saveString(ofstream& file, const string& str)
// Algorithm : Save a string to a file
{
    int size = str.size();
    file.write(reinterpret_cast<const char*>(&size), sizeof(int));
    file.write(str.data(), size);
}  //----- End of saveString


void NNAI::loadLayers(ifstream &file)
// Algorithm : Load the parameters of a fully connected layer from a file
// and add the layers to the FCNN
{
    // Load layer type
    string layerType;
    loadString(file, layerType);

    // Load dimensions
    int inputSize, outputSize;
    file.read(reinterpret_cast<char*>(&inputSize), sizeof(inputSize));
    file.read(reinterpret_cast<char*>(&outputSize), sizeof(outputSize));

    // Add layer to network
    aiFCNN.addLayer(layerType, outputSize, inputSize);

    // Load weights
    Matrix<float> tmpWeights(inputSize, outputSize);
    for(auto &row : tmpWeights) {
        for(float &w : row) {
            file.read(reinterpret_cast<char*>(&w), sizeof(w));
        }
    }
    aiFCNN.getNN().back().weights = tmpWeights;

    // Load biases
    vector<float> tmpBiases(outputSize, 0.0);  // initialise bias to 0
    for(float &b : tmpBiases) {
        file.read(reinterpret_cast<char*>(&b), sizeof(b));
    }
    aiFCNN.getNN().back().biases = tmpBiases;
}  //----- End of loadLayers


void NNAI::loadConvLayers(ifstream &file)
// Algorithm : Load the parameters of a convolutional layer from a file
// and add the layers to the CNN
{
    // Load layer parameters
    string layerType;
    loadString(file, layerType);

    int kernelSize, stride, padding, outputChannels, inputChannels;
    file.read(reinterpret_cast<char*>(&kernelSize), sizeof(kernelSize));
    file.read(reinterpret_cast<char*>(&stride), sizeof(stride));
    file.read(reinterpret_cast<char*>(&padding), sizeof(padding));
    file.read(reinterpret_cast<char*>(&outputChannels), sizeof(outputChannels));
    file.read(reinterpret_cast<char*>(&inputChannels), sizeof(inputChannels));

    // Compute the output size with padding
    pair<int, int> outputSize;
    if (aiCNN.getNNSize() == 0)  // First layer
    {
        outputSize = {(config.H + 2*padding - kernelSize)/stride + 1, (config.W + 2*padding - kernelSize)/stride + 1};
    }
    else
    {
        outputSize = {(aiCNN.getNN().back().outputSize.first + 2*padding - kernelSize)/stride + 1,
                         (aiCNN.getNN().back().outputSize.second + 2*padding - kernelSize)/stride + 1};
    }

    // Add layer to network
    aiCNN.addLayer(layerType, kernelSize, stride, padding, outputChannels, outputSize, inputChannels);

    // Load weights and biases
    Matrix<Matrix<float>> tmpWeights(inputChannels, outputChannels,
        Matrix<float>(kernelSize, kernelSize, 0.0));
    vector<float> tmpBiases(outputChannels, 0.0);
    for(int ic = 0; ic < inputChannels; ic++) {
        for(int oc = 0; oc < outputChannels; oc++) {
            for(int kh = 0; kh < kernelSize; kh++) {
                for(int kw = 0; kw < kernelSize; kw++) {
                    file.read(reinterpret_cast<char*>(&tmpWeights[ic][oc][kh][kw]),
                             sizeof(float));
                }
            }
        }
    }

    for(int oc = 0; oc < outputChannels; oc++) {
        file.read(reinterpret_cast<char*>(&tmpBiases[oc]), sizeof(float));
    }
    aiCNN.getNN().back().weights = tmpWeights;
    aiCNN.getNN().back().biases = tmpBiases;
}  //----- End of loadConvLayers


void NNAI::loadVector(ifstream& file, vector<float>& vec)
// Algorithm : Load a vector of floats from a file
{
    int size;
    file.read(reinterpret_cast<char*>(&size), sizeof(int));

    vec.resize(size);
    if(size > 0) {
        file.read(reinterpret_cast<char*>(vec.data()), size * sizeof(float));
    }
}  //----- End of loadVector


void NNAI::loadString(ifstream& file, string& str)
// Algorithm : Load a string from a file
{
    int size;
    file.read(reinterpret_cast<char*>(&size), sizeof(int));

    str.resize(size);
    if(size > 0) {
        file.read(&str[0], size);
    }
}  //----- End of loadString


inline void NNAI::decayEpsilon()
// Algorithm : Decay epsilon to gradually shift from exploration to exploitation
{
    if (epsilon > epsilonMin)
    {
        epsilon *= epsilonDecay;
    }
}  //----- End of decayEpsilon


vector<float> combineFeatures(const vector<float> &boardState, const vector<float> &extras)
// Algorithm : Combine the board state and extra features into a single vector
{
    vector<float> combined(boardState.size() + extras.size());
    copy(boardState.begin(), boardState.end(), combined.begin());
    copy(extras.begin(), extras.end(), combined.begin() + boardState.size());

    return combined;
}  //----- End of combineFeatures