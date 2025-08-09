/*************************************************************************
ReplayBuffer - A replay buffer for the Deep Q-Learning algorithm
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//- Implementation of the module <ReplayBuffer> (file ReplayBuffer.cpp) -

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <iostream>
#include <algorithm>

using namespace std;

//------------------------------------------------------- Personal Include
#include "../include/ReplayBuffer.h"

//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
void ReplayBuffer::push(const Transition &transition)
// Algorithm : Add a transition to the buffer and remove
// 10% of the lowest priority ones if the buffer is full
{
    if(buffer.size() >= capacity)
  {
        removeLowPriority(0.1*buffer.size(), 0.2);  // remove 10% of the buffer
    }

    buffer.push_back(transition);
}  //----- End of push


Transition ReplayBuffer::createTransition(const vector<float> &currentBoardState,
                                          const vector<float> &currentExtraState, const vector<float> &nextBoardState,
                                          const vector<float> &nextExtraState, const string &bestMove, float reward, bool done)
// Algorithm : Creates a transition
{
    // --- Create and store transition ---
    Transition transition;
    transition.boardState = currentBoardState;
    transition.extraState = currentExtraState;
    transition.action = bestMove;
    transition.reward = reward;
    transition.nextBoardState = nextBoardState;
    transition.nextExtraState = nextExtraState;
    transition.done = done;
    transition.tdError = 0.0;

    return transition;
}  // ----- End of createTransition


void ReplayBuffer::mergeSamples(vector<Transition> &top, vector<Transition> &rest, vector<Transition> &output)
// Algorithm : Merges the top and rest samples and
// shuffles them locally while keeping a 1:2 ratio
{
    const int GROUP_SIZE = 3; // 1 top + 2 rest
    const int TOTAL_GROUPS = output.capacity() / GROUP_SIZE;

    for(int i = 0; i < TOTAL_GROUPS; i++)
    {
        if(!top.empty())
        {
            output.push_back(top.back());
            top.pop_back();
        }

        for(int j = 0; j < 2 && !rest.empty(); j++)
        {
            output.push_back(rest.back());
            rest.pop_back();
        }
    }

    shuffle(output.begin(), output.end(), gen);
}  // ----- End of mergeSamples


vector<Transition> ReplayBuffer::sample(int start, int end, int batchSize)
// Algorithm : Samples batchSize elements from the buffer between
// start and end indices. When [start, end] > batchSize, elements
// are randomly replaced to ensure diversity
{
    vector<Transition> samples;
    if(start < end)
    {
        for(int i = start; i < end; i++)
        {
            if(samples.size() < batchSize)
            {
                samples.push_back(buffer[i]);
            }
            else
            {
                // randomly replace elements
                uniform_int_distribution<int> dist(0, i);
                int j = dist(gen);

                if(j < batchSize)
                {
                    samples[j] = buffer[i];
                }
            }
        }
    }

    return samples;
}  // ----- End of sample


vector<Transition> ReplayBuffer::createBatch(int batchSize)
// Algorithm : Creates a batch with a mix of high and low
// priority transitions. The batch is shuffled and noise is
// injected to increase diversity
{
    vector<Transition> batch;
    batch.reserve(batchSize);

    // dynamic partition point
    const int partitionPoint = min(
        static_cast<size_t>(buffer.size() * 0.15),
        buffer.size() - batchSize / 3
    );

    // Stratified sampling
    vector<Transition> topSamples = sample(0, partitionPoint, batchSize / 3);  // 1/3 with high priority transitions
    vector<Transition> restSamples = sample(partitionPoint, buffer.size(), 2 * batchSize / 3);  // 2/3

    // shuffle samples with partial preservation of the order
    mergeSamples(topSamples, restSamples, batch);

    // dynamic diversity management
    entropyInjection(batch, 0.1); // 10% of noise

    return batch;
}


void ReplayBuffer::entropyInjection(vector<Transition>& batch, float entropyWeight)
// Algorithm : Injects noise in the batch by randomly
// replacing elements to increase diversity
{
    const int noiseCount = batch.size() * entropyWeight;
    uniform_int_distribution<int> dist(0, buffer.size() - 1);

    for(int i = 0; i < noiseCount; i++)
    {
        int randomInd = dist(gen);

        // Randomly replace an element in the batch
        batch[uniform_int_distribution<int>(0, batch.size()-1)(gen)] = buffer[randomInd];
    }
}  // ----- End of entropyInjection


void ReplayBuffer::removeLowPriority(int numToRemove, float priorityThreshold)
// Algorithm : Removes the lowest priority transitions from the buffer
// based on a threshold computed from the TD error
{
    if (buffer.empty())
    {
        return;
    }

    // dynamic thresholding based on TD error
    auto [minIt, maxIt] = minmax_element(buffer.begin(), buffer.end(),
        [](const Transition& a, const Transition& b) {
            return a.tdError < b.tdError;
        });

    const float MIN_ERR = minIt->tdError;
    const float MAX_ERR = maxIt->tdError;
    const float THRESHOLD = MIN_ERR + (MAX_ERR - MIN_ERR) * priorityThreshold;

    // partition the buffer based on the threshold
    auto newEnd = partition(buffer.begin(), buffer.end(),
        [THRESHOLD](const Transition& t) {
            return t.tdError > THRESHOLD;
        });

    const int REMAINING = distance(buffer.begin(), newEnd);

    // ensure that we don't remove more elements than necessary
    if (REMAINING < buffer.size() - numToRemove)
    {
        newEnd = buffer.begin() + (buffer.size() - numToRemove);
    }

    buffer.erase(newEnd, buffer.end());
}  // ----- End of removeLowPriority